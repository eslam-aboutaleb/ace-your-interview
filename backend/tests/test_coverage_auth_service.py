"""Coverage for ``app.services.auth`` — allowed-users store, JWT, OAuth exchange.

The OAuth exchanges are driven with a fake ``httpx``-shaped client injected
through ``get_oauth_http_client`` so no socket is ever opened.
"""

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import jwt

from app.config import get_settings
from app.services import auth as auth_service
from app.services.auth import (
    add_allowed_user,
    create_jwt_token,
    decode_jwt_token,
    exchange_github_code,
    exchange_google_code,
    get_allowed_users,
    get_allowed_users_file,
    is_allowed,
    remove_allowed_user,
)

STRONG_SECRET = "s" * 64


class _FakeResponse:
    def __init__(self, payload=None, text="", status_code=200, raises=None):
        self._payload = payload if payload is not None else {}
        self.text = text
        self.status_code = status_code
        self._raises = raises

    def raise_for_status(self):
        if self._raises is not None:
            raise self._raises

    def json(self):
        return self._payload


class _RecordingOAuthClient:
    """Captures the outbound calls the exchange helpers make."""

    def __init__(self, token_response=None, user_response=None):
        self.token_response = token_response or _FakeResponse({"access_token": "tok"})
        self.user_response = user_response or _FakeResponse({"login": "octocat"})
        self.calls: list[tuple[str, str, dict]] = []

    async def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.token_response

    async def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.user_response


class AllowedUsersStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.allowed_file = Path(self.tmpdir.name) / "allowed_users.json"
        self.prev_env = {
            key: os.environ.get(key)
            for key in ("STUDY_ALLOWED_USERS_FILE", "STUDY_ALLOWED_GITHUB_USERS", "STUDY_ALLOWED_GOOGLE_EMAILS")
        }
        os.environ["STUDY_ALLOWED_USERS_FILE"] = str(self.allowed_file)
        os.environ["STUDY_ALLOWED_GITHUB_USERS"] = " Seeded-User , ghost "
        os.environ["STUDY_ALLOWED_GOOGLE_EMAILS"] = "Seeded@Example.com,"
        get_settings.cache_clear()

    def tearDown(self):
        self.tmpdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def test_allowed_users_file_is_resolved_from_settings(self):
        self.assertEqual(get_allowed_users_file(), self.allowed_file)

    def test_first_load_seeds_from_env_and_persists_normalised_values(self):
        self.assertFalse(self.allowed_file.exists())
        data = get_allowed_users()

        self.assertEqual(data["github_users"], ["seeded-user", "ghost"])
        self.assertEqual(data["google_emails"], ["seeded@example.com"])
        # The seeded data must be written so the next load no longer needs env vars.
        self.assertTrue(self.allowed_file.exists())
        with open(self.allowed_file) as f:
            self.assertEqual(json.load(f), data)

    def test_existing_file_short_circuits_the_env_seed(self):
        with open(self.allowed_file, "w") as f:
            json.dump({"github_users": ["from-file"], "google_emails": []}, f)

        os.environ["STUDY_ALLOWED_GITHUB_USERS"] = "from-env"
        get_settings.cache_clear()

        self.assertEqual(get_allowed_users()["github_users"], ["from-file"])

    def test_add_allowed_user_appends_normalised_identifier_per_provider(self):
        data = add_allowed_user("github", "  NewUser ")
        self.assertIn("newuser", data["github_users"])
        # A google identifier must not leak into the github bucket.
        self.assertNotIn("newuser", data["google_emails"])

        data = add_allowed_user("google", "Person@Example.com")
        self.assertIn("person@example.com", data["google_emails"])

        with open(self.allowed_file) as f:
            self.assertEqual(json.load(f), data)

    def test_add_allowed_user_is_idempotent_for_the_same_identity(self):
        add_allowed_user("github", "dup")
        before = get_allowed_users()["github_users"].count("dup")
        data = add_allowed_user("github", "DUP")

        self.assertEqual(before, 1)
        self.assertEqual(data["github_users"].count("dup"), 1)

    def test_add_allowed_user_ignores_an_unknown_provider(self):
        data = add_allowed_user("gitlab", "someone")
        self.assertNotIn("someone", data["github_users"])
        self.assertNotIn("someone", data["google_emails"])

    def test_remove_allowed_user_filters_the_matching_provider_only(self):
        add_allowed_user("github", "keep")
        add_allowed_user("github", "drop")
        add_allowed_user("google", "drop")

        data = remove_allowed_user("github", " Drop ")
        self.assertIn("keep", data["github_users"])
        self.assertNotIn("drop", data["github_users"])
        self.assertIn("drop", data["google_emails"])

        data = remove_allowed_user("google", "drop")
        self.assertNotIn("drop", data["google_emails"])

        with open(self.allowed_file) as f:
            self.assertEqual(json.load(f), data)

    def test_remove_allowed_user_is_a_no_op_for_an_unknown_provider(self):
        add_allowed_user("github", "keep")
        data = remove_allowed_user("gitlab", "keep")
        self.assertIn("keep", data["github_users"])

    def test_is_allowed_covers_both_providers_and_the_unknown_case(self):
        add_allowed_user("github", "octocat")
        add_allowed_user("google", "octo@example.com")

        self.assertTrue(is_allowed("github", " OctoCat "))
        self.assertTrue(is_allowed("google", "OCTO@EXAMPLE.COM"))
        self.assertFalse(is_allowed("github", "octo@example.com"))
        self.assertFalse(is_allowed("gitlab", "octocat"))


class JwtHelperTests(unittest.TestCase):
    def setUp(self):
        self.prev_secret = os.environ.get("STUDY_AUTH_SECRET_KEY")
        os.environ["STUDY_AUTH_SECRET_KEY"] = STRONG_SECRET
        get_settings.cache_clear()

    def tearDown(self):
        if self.prev_secret is None:
            os.environ.pop("STUDY_AUTH_SECRET_KEY", None)
        else:
            os.environ["STUDY_AUTH_SECRET_KEY"] = self.prev_secret
        get_settings.cache_clear()

    def test_round_trip_preserves_subject_and_provider(self):
        token = create_jwt_token("alice", "github")
        payload = decode_jwt_token(token)
        self.assertEqual(payload["sub"], "alice")
        self.assertEqual(payload["provider"], "github")
        self.assertEqual(payload, {"sub": "alice", "provider": "github"})

    def test_token_is_hs256_and_signed_with_the_configured_secret(self):
        token = create_jwt_token("alice", "github")
        header = jwt.get_unverified_header(token)
        self.assertEqual(header["alg"], "HS256")
        self.assertEqual(
            jwt.decode(token, STRONG_SECRET, algorithms=["HS256"])["sub"], "alice"
        )

    def test_expired_token_is_rejected(self):
        expired = jwt.encode(
            {
                "sub": "alice",
                "provider": "github",
                "exp": int(time.time()) - 60,
            },
            STRONG_SECRET,
            algorithm="HS256",
        )
        with self.assertRaises(jwt.ExpiredSignatureError):
            decode_jwt_token(expired)

    def test_token_signed_with_a_different_secret_is_rejected(self):
        forged = jwt.encode(
            {"sub": "mallory", "provider": "github"}, "o" * 64, algorithm="HS256"
        )
        with self.assertRaises(jwt.InvalidSignatureError):
            decode_jwt_token(forged)

    def test_alg_none_token_is_rejected(self):
        unsigned = jwt.encode(
            {"sub": "mallory", "provider": "github"}, key="", algorithm="none"
        )
        with self.assertRaises(jwt.InvalidAlgorithmError):
            decode_jwt_token(unsigned)

    def test_malformed_token_is_rejected(self):
        for token in ("not-a-jwt", "", "a.b.c"):
            with self.subTest(token=token):
                with self.assertRaises(jwt.PyJWTError):
                    decode_jwt_token(token)

    def test_weak_signing_secret_is_refused_at_encode_time(self):
        os.environ["STUDY_AUTH_SECRET_KEY"] = "test"
        get_settings.cache_clear()
        with self.assertRaises(RuntimeError) as ctx:
            create_jwt_token("alice", "github")
        self.assertIn("STUDY_AUTH_SECRET_KEY", str(ctx.exception))


class GithubExchangeTests(unittest.TestCase):
    def setUp(self):
        self.prev_env = {
            key: os.environ.get(key)
            for key in ("STUDY_GITHUB_CLIENT_ID", "STUDY_GITHUB_CLIENT_SECRET")
        }
        os.environ["STUDY_GITHUB_CLIENT_ID"] = "gh-id"
        os.environ["STUDY_GITHUB_CLIENT_SECRET"] = "gh-secret"
        get_settings.cache_clear()

    def tearDown(self):
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def test_exchange_returns_the_login_and_posts_the_client_credentials(self):
        client = _RecordingOAuthClient(
            token_response=_FakeResponse({"access_token": "gho_x"}),
            user_response=_FakeResponse({"login": "OctoCat"}),
        )
        with patch.object(auth_service, "get_oauth_http_client", return_value=client):
            login = asyncio.run(exchange_github_code("code-1"))

        self.assertEqual(login, "OctoCat")
        methods = [call[0] for call in client.calls]
        self.assertEqual(methods, ["POST", "GET"])
        _, post_url, post_kwargs = client.calls[0]
        self.assertEqual(post_url, "https://github.com/login/oauth/access_token")
        self.assertEqual(
            post_kwargs["json"],
            {
                "client_id": "gh-id",
                "client_secret": "gh-secret",
                "code": "code-1",
            },
        )
        _, get_url, get_kwargs = client.calls[1]
        self.assertEqual(get_url, "https://api.github.com/user")
        self.assertEqual(get_kwargs["headers"]["Authorization"], "Bearer gho_x")

    def test_response_without_access_token_raises_value_error(self):
        client = _RecordingOAuthClient(token_response=_FakeResponse({"error": "bad_code"}))
        with patch.object(auth_service, "get_oauth_http_client", return_value=client):
            with self.assertRaises(ValueError) as ctx:
                asyncio.run(exchange_github_code("code-1"))

        self.assertIn("GitHub token error", str(ctx.exception))
        self.assertIn("bad_code", str(ctx.exception))
        # No user lookup may be attempted without a token.
        self.assertEqual([call[0] for call in client.calls], ["POST"])

    def test_http_error_on_token_request_propagates(self):
        client = _RecordingOAuthClient(
            token_response=_FakeResponse(raises=RuntimeError("500 from github"))
        )
        with patch.object(auth_service, "get_oauth_http_client", return_value=client):
            with self.assertRaises(RuntimeError):
                asyncio.run(exchange_github_code("code-1"))

    def test_http_error_on_user_lookup_propagates(self):
        client = _RecordingOAuthClient(
            user_response=_FakeResponse(raises=RuntimeError("401 from github"))
        )
        with patch.object(auth_service, "get_oauth_http_client", return_value=client):
            with self.assertRaises(RuntimeError):
                asyncio.run(exchange_github_code("code-1"))


class GoogleExchangeTests(unittest.TestCase):
    def setUp(self):
        self.prev_env = {
            key: os.environ.get(key)
            for key in ("STUDY_GOOGLE_CLIENT_ID", "STUDY_GOOGLE_CLIENT_SECRET")
        }
        os.environ["STUDY_GOOGLE_CLIENT_ID"] = "goog-id"
        os.environ["STUDY_GOOGLE_CLIENT_SECRET"] = "goog-secret"
        get_settings.cache_clear()

    def tearDown(self):
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def test_exchange_returns_the_email_and_posts_a_form_encoded_grant(self):
        client = _RecordingOAuthClient(
            token_response=_FakeResponse({"access_token": "ya29.x"}),
            user_response=_FakeResponse({"email": "user@example.com"}),
        )
        with patch.object(auth_service, "get_oauth_http_client", return_value=client):
            email = asyncio.run(
                exchange_google_code("code-1", "https://app.test/api/auth/google/callback")
            )

        self.assertEqual(email, "user@example.com")
        _, post_url, post_kwargs = client.calls[0]
        self.assertEqual(post_url, "https://oauth2.googleapis.com/token")
        self.assertEqual(post_kwargs["data"]["grant_type"], "authorization_code")
        self.assertEqual(post_kwargs["data"]["client_id"], "goog-id")
        self.assertEqual(
            post_kwargs["data"]["redirect_uri"],
            "https://app.test/api/auth/google/callback",
        )
        _, get_url, get_kwargs = client.calls[1]
        self.assertEqual(get_url, "https://www.googleapis.com/oauth2/v2/userinfo")
        self.assertEqual(get_kwargs["headers"]["Authorization"], "Bearer ya29.x")

    def test_missing_access_token_key_raises_keyerror(self):
        client = _RecordingOAuthClient(token_response=_FakeResponse({"id_token": "x"}))
        with patch.object(auth_service, "get_oauth_http_client", return_value=client):
            with self.assertRaises(KeyError):
                asyncio.run(exchange_google_code("code-1", "https://app.test/cb"))

    def test_http_error_on_token_request_propagates(self):
        client = _RecordingOAuthClient(
            token_response=_FakeResponse(raises=RuntimeError("400 invalid_grant"))
        )
        with patch.object(auth_service, "get_oauth_http_client", return_value=client):
            with self.assertRaises(RuntimeError):
                asyncio.run(exchange_google_code("code-1", "https://app.test/cb"))


if __name__ == "__main__":
    unittest.main()