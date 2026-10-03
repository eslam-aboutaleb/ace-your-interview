"""Coverage for ``app.services.user_settings_store``.

This store owns per-user LLM preferences plus the encrypted credentials and
Google OAuth tokens behind them. The tests are organised around the trust
boundaries: preference sanitisation (everything read from the file is
untrusted), credential encryption/decryption failure handling, Google token
refresh (HTTP is always stubbed), and the policy resolution that decides whether
a call may proceed.
"""

import asyncio
import contextlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from app.config import get_settings
from app.services.llm_assignments_store import LLMAssignmentsStore
from app.services.llm_policy import (
    APPROVAL_REQUIRED_CODE,
    PERSONAL_CREDENTIAL_REQUIRED_CODE,
    STUDY_APP_NOT_ASSIGNED_CODE,
)
from app.services.llm_service_access import LLMServiceAccess
from app.services.user_settings_store import (
    UserSettingsStore,
    identity_key_for_user,
    resolve_credentials_encryption_secret,
)

_IDENTITY = "google:learner@example.com"


def _fail_stat_for(target):
    """Return a ``Path.stat`` replacement that raises OSError only for ``target``."""

    real_stat = Path.stat

    def _stat(self, *args, **kwargs):
        if str(self) == target:
            raise OSError("stat boom")
        return real_stat(self, *args, **kwargs)

    return _stat


def _fake_response(payload, *, error=None):
    if error is not None:
        raise error

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    return _Resp()


class _FakeOAuthClient:
    """Stub for ``get_oauth_http_client``; never opens a socket."""

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls: list[dict] = []

    async def post(self, url, data=None, timeout=None):
        self.calls.append({"url": url, "data": data, "timeout": timeout})
        return _fake_response(self.payload, error=self.error)


class _StoreFixture(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.user_settings_file = os.path.join(self.tmpdir.name, "user_settings.json")
        self.prev_env = {
            "STUDY_USER_SETTINGS_FILE": os.environ.get("STUDY_USER_SETTINGS_FILE"),
            "STUDY_LLM_SERVICE_USERS_FILE": os.environ.get("STUDY_LLM_SERVICE_USERS_FILE"),
            "STUDY_LLM_ASSIGNMENTS_FILE": os.environ.get("STUDY_LLM_ASSIGNMENTS_FILE"),
            "STUDY_CREDENTIALS_ENCRYPTION_KEY": os.environ.get("STUDY_CREDENTIALS_ENCRYPTION_KEY"),
            "STUDY_GOOGLE_CLIENT_ID": os.environ.get("STUDY_GOOGLE_CLIENT_ID"),
            "STUDY_GOOGLE_CLIENT_SECRET": os.environ.get("STUDY_GOOGLE_CLIENT_SECRET"),
            "STUDY_ADMIN_USERS": os.environ.get("STUDY_ADMIN_USERS"),
            "STUDY_DEFAULT_PROVIDER": os.environ.get("STUDY_DEFAULT_PROVIDER"),
        }
        os.environ["STUDY_USER_SETTINGS_FILE"] = self.user_settings_file
        os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = os.path.join(self.tmpdir.name, "users.json")
        os.environ["STUDY_LLM_ASSIGNMENTS_FILE"] = os.path.join(self.tmpdir.name, "assignments.json")
        os.environ["STUDY_CREDENTIALS_ENCRYPTION_KEY"] = "coverage-encryption-secret"
        os.environ["STUDY_GOOGLE_CLIENT_ID"] = "cid"
        os.environ["STUDY_GOOGLE_CLIENT_SECRET"] = "csecret"
        os.environ["STUDY_ADMIN_USERS"] = ""
        get_settings.cache_clear()
        self.access = LLMServiceAccess()
        self.assignments = LLMAssignmentsStore()
        self.store = UserSettingsStore(
            llm_service_access=self.access,
            llm_assignments_store=self.assignments,
        )

    def tearDown(self):
        self.tmpdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _write_raw(self, payload):
        with open(self.user_settings_file, "w", encoding="utf-8") as handle:
            if isinstance(payload, str):
                handle.write(payload)
            else:
                json.dump(payload, handle)

    def _write_record(self, record, identity=_IDENTITY):
        self._write_raw({"users": {identity: record}})

    def _oauth_record(self, identity=_IDENTITY):
        with open(self.user_settings_file, encoding="utf-8") as handle:
            return json.load(handle)["users"][identity]["google_oauth"]

    @contextlib.contextmanager
    def _store_file_stat_fails(self):
        """Make the settings file visible but un-``stat``-able."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat", new=_fail_stat_for(self.user_settings_file)),
        ):
            yield


class IdentityKeyTests(unittest.TestCase):
    def test_identity_key_is_provider_and_lowercased_user(self):
        self.assertEqual(
            identity_key_for_user({"provider": " Google ", "user": " Learner@Example.COM "}),
            "google:learner@example.com",
        )

    def test_absent_or_empty_identity_yields_empty_key(self):
        for identity in (None, {}, {"provider": "google"}, {"user": "   "}, {"user": ""}):
            with self.subTest(identity=identity):
                self.assertEqual(identity_key_for_user(identity), "")

    def test_non_string_user_is_stringified(self):
        # ``str(user_identity["user"])`` means a null subject becomes the literal
        # identity ":none" rather than being rejected.
        self.assertEqual(identity_key_for_user({"provider": "google", "user": None}), "google:none")

    def test_missing_provider_yields_bare_user_key(self):
        self.assertEqual(identity_key_for_user({"user": "a@x.com"}), ":a@x.com")


class EncryptionSecretTests(unittest.TestCase):
    def test_explicit_key_wins(self):
        class _S:
            credentials_encryption_key = "  explicit  "
            environment = "production"
            allow_insecure_dev_encryption_fallback = False
            insecure_dev_credentials_encryption_secret = ""

        self.assertEqual(resolve_credentials_encryption_secret(_S()), "explicit")

    def test_blank_key_with_fallback_enabled_but_no_secret_raises(self):
        class _S:
            credentials_encryption_key = "   "
            environment = "development"
            allow_insecure_dev_encryption_fallback = True
            insecure_dev_credentials_encryption_secret = ""

        with self.assertRaisesRegex(RuntimeError, "INSECURE_DEV_CREDENTIALS_ENCRYPTION_SECRET"):
            resolve_credentials_encryption_secret(_S())

    def test_development_fallback_secret_is_accepted_when_explicitly_allowed(self):
        class _S:
            credentials_encryption_key = ""
            environment = "development"
            allow_insecure_dev_encryption_fallback = True
            insecure_dev_credentials_encryption_secret = "  dev-secret  "

        self.assertEqual(resolve_credentials_encryption_secret(_S()), "dev-secret")

    def test_fallback_is_refused_outside_development(self):
        class _S:
            credentials_encryption_key = ""
            environment = "production"
            allow_insecure_dev_encryption_fallback = True
            insecure_dev_credentials_encryption_secret = "dev-secret"

        with self.assertRaisesRegex(RuntimeError, "CREDENTIALS_ENCRYPTION_KEY"):
            resolve_credentials_encryption_secret(_S())


class IdentityPartsTests(unittest.TestCase):
    def test_provider_scoped_key_is_split(self):
        self.assertEqual(
            UserSettingsStore._identity_parts("Google:Learner@Example.com"),
            ("google", "learner@example.com"),
        )

    def test_key_without_provider_has_empty_provider(self):
        self.assertEqual(UserSettingsStore._identity_parts("learner@example.com"), ("", "learner@example.com"))

    def test_blank_key_yields_both_parts_empty(self):
        for key in ("", "   ", None):
            with self.subTest(key=key):
                self.assertEqual(UserSettingsStore._identity_parts(key), ("", ""))

    def test_only_the_first_colon_splits(self):
        self.assertEqual(UserSettingsStore._identity_parts("oidc:a:b"), ("oidc", "a:b"))


class StaticHelpersTests(_StoreFixture):
    def test_providers_returns_a_copy_of_the_whitelist(self):
        providers = UserSettingsStore.providers()
        self.assertEqual(set(providers), {"google", "openai", "anthropic", "groq"})
        providers.append("mutated")
        self.assertNotIn("mutated", UserSettingsStore.providers())

    def test_provider_models_lists_models_per_whitelisted_provider(self):
        models = self.store.provider_models()
        self.assertEqual(set(models), {"google", "openai", "anthropic", "groq"})
        self.assertIn("gpt-4o-mini", models["openai"])

    def test_default_preferences_follow_the_configured_provider(self):
        defaults = self.store.default_preferences()
        self.assertEqual(defaults["provider"], "groq")
        self.assertEqual(defaults["model"], "llama-3.3-70b-versatile")
        self.assertEqual(defaults["auth_mode"], "api_key")
        self.assertEqual(defaults["llm_source"], "personal")
        self.assertEqual(defaults["max_tokens"], 0)
        self.assertIsNone(defaults["voice_tier"])

    def test_default_preferences_fall_back_to_openai_for_unwhitelisted_provider(self):
        os.environ["STUDY_DEFAULT_PROVIDER"] = "ollama"
        get_settings.cache_clear()
        store = UserSettingsStore(
            llm_service_access=self.access,
            llm_assignments_store=self.assignments,
        )
        self.assertEqual(store.default_preferences()["provider"], "openai")


class PreferenceSanitisationTests(_StoreFixture):
    def _sanitise(self, raw, defaults=None):
        return UserSettingsStore._sanitise_preferences(raw, defaults or self.store.default_preferences())

    def test_unknown_provider_falls_back_to_the_default(self):
        out = self._sanitise({"provider": "ollama", "model": "llama3.2"})
        self.assertEqual(out["provider"], "groq")
        self.assertEqual(out["model"], "")

    def test_model_outside_the_provider_catalogue_is_cleared(self):
        out = self._sanitise({"provider": "openai", "model": "gpt-9-imaginary"})
        self.assertEqual(out["provider"], "openai")
        self.assertEqual(out["model"], "")

    def test_known_model_is_kept(self):
        out = self._sanitise({"provider": "openai", "model": " gpt-4o-mini "})
        self.assertEqual(out["model"], "gpt-4o-mini")

    def test_unknown_auth_mode_falls_back_to_api_key(self):
        out = self._sanitise({"provider": "openai", "auth_mode": "oauth"})
        self.assertEqual(out["auth_mode"], "api_key")

    def test_account_auth_mode_is_forced_to_api_key_for_non_google(self):
        out = self._sanitise({"provider": "openai", "auth_mode": "account"})
        self.assertEqual(out["auth_mode"], "api_key")

    def test_account_auth_mode_survives_for_google(self):
        out = self._sanitise({"provider": "google", "auth_mode": "account"})
        self.assertEqual(out["provider"], "google")
        self.assertEqual(out["auth_mode"], "account")

    def test_unknown_llm_source_falls_back_to_personal(self):
        out = self._sanitise({"llm_source": "study-app"})
        self.assertEqual(out["llm_source"], "personal")

    def test_study_app_llm_source_is_kept(self):
        out = self._sanitise({"llm_source": " Study_App "})
        self.assertEqual(out["llm_source"], "study_app")

    def test_non_numeric_temperature_falls_back_to_the_default(self):
        out = self._sanitise({"temperature": "hot"})
        self.assertAlmostEqual(out["temperature"], 0.7)

    def test_temperature_is_clamped_to_the_supported_range(self):
        self.assertEqual(self._sanitise({"temperature": 5})["temperature"], 2.0)
        self.assertEqual(self._sanitise({"temperature": -1})["temperature"], 0.0)

    def test_non_numeric_max_tokens_falls_back_to_the_default(self):
        out = self._sanitise({"max_tokens": "lots"})
        self.assertEqual(out["max_tokens"], 0)

    def test_negative_max_tokens_is_clamped_to_zero(self):
        self.assertEqual(self._sanitise({"max_tokens": -50})["max_tokens"], 0)

    def test_require_answer_reveal_accepts_bool_int_float_and_strings(self):
        cases = [
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            (2.5, True),
            (0.0, False),
            ("yes", True),
            (" ON ", True),
            ("1", True),
            ("off", False),
            ("no", False),
            ("0", False),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertIs(self._sanitise({"require_answer_reveal": raw})["require_answer_reveal"], expected)

    def test_unrecognised_string_uses_the_default(self):
        out = self._sanitise({"require_answer_reveal": "maybe"})
        self.assertIs(out["require_answer_reveal"], False)

    def test_unsupported_type_uses_the_default(self):
        for raw in ([], {}, None):
            with self.subTest(raw=raw):
                self.assertIs(self._sanitise({"require_answer_reveal": raw})["require_answer_reveal"], False)

    def test_unknown_voice_tier_is_dropped(self):
        out = self._sanitise({"voice_tier": "ultra"})
        self.assertIsNone(out["voice_tier"])

    def test_known_voice_tiers_survive(self):
        self.assertEqual(self._sanitise({"voice_tier": " cloud "})["voice_tier"], "cloud")
        self.assertEqual(self._sanitise({"voice_tier": "browser"})["voice_tier"], "browser")

    def test_missing_keys_take_the_defaults(self):
        out = self._sanitise({})
        self.assertEqual(out, self.store.default_preferences())


class GetUserStateTests(_StoreFixture):
    def test_empty_identity_returns_defaults_without_touching_disk(self):
        prefs, has_saved = self.store.get_user_state("")
        self.assertFalse(has_saved)
        self.assertEqual(prefs, self.store.default_preferences())

    def test_unknown_identity_returns_defaults(self):
        prefs, has_saved = self.store.get_user_state("google:nobody@example.com")
        self.assertFalse(has_saved)
        self.assertEqual(prefs, self.store.default_preferences())

    def test_saved_preferences_are_returned_and_sanitised(self):
        self.store.save_preferences(
            _IDENTITY,
            {"provider": "openai", "model": "gpt-4o", "temperature": 0.4, "llm_source": "study_app"},
        )
        prefs, has_saved = self.store.get_user_state(_IDENTITY)
        self.assertTrue(has_saved)
        self.assertEqual(prefs["provider"], "openai")
        self.assertEqual(prefs["model"], "gpt-4o")
        self.assertAlmostEqual(prefs["temperature"], 0.4)
        self.assertEqual(prefs["llm_source"], "study_app")

    def test_record_without_preferences_falls_back_to_defaults(self):
        self._write_record({"credentials": {}, "google_oauth": {}})
        prefs, has_saved = self.store.get_user_state(_IDENTITY)
        self.assertFalse(has_saved)
        self.assertEqual(prefs, self.store.default_preferences())

    def test_non_dict_preferences_fall_back_to_defaults(self):
        self._write_record({"preferences": "not-a-dict"})
        prefs, has_saved = self.store.get_user_state(_IDENTITY)
        self.assertFalse(has_saved)
        self.assertEqual(prefs, self.store.default_preferences())

    def test_non_dict_record_falls_back_to_defaults(self):
        self._write_record("not-a-dict")
        prefs, has_saved = self.store.get_user_state(_IDENTITY)
        self.assertFalse(has_saved)
        self.assertEqual(prefs, self.store.default_preferences())


class SavePreferencesTests(_StoreFixture):
    def test_empty_identity_returns_sanitised_prefs_without_persisting(self):
        clean = self.store.save_preferences("", {"provider": "openai", "model": "gpt-4o"})
        self.assertEqual(clean["provider"], "openai")
        self.assertFalse(os.path.exists(self.user_settings_file))

    def test_save_creates_the_credential_and_oauth_sections(self):
        self.store.save_preferences(_IDENTITY, {"provider": "openai", "model": "gpt-4o"})
        with open(self.user_settings_file, encoding="utf-8") as handle:
            raw = json.load(handle)["users"][_IDENTITY]
        self.assertEqual(raw["credentials"], {})
        self.assertEqual(raw["google_oauth"], {})
        self.assertEqual(raw["preferences"]["model"], "gpt-4o")

    def test_save_preserves_existing_credentials(self):
        self.store.set_api_key(_IDENTITY, "openai", "sk-user")
        self.store.save_preferences(_IDENTITY, {"provider": "openai", "model": "gpt-4o"})
        self.assertEqual(self.store.get_api_key(_IDENTITY, "openai"), "sk-user")


class ApiKeyTests(_StoreFixture):
    def test_unsupported_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported provider"):
            self.store.set_api_key(_IDENTITY, "ollama", "sk-user")

    def test_missing_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing identity"):
            self.store.set_api_key("", "openai", "sk-user")

    def test_blank_api_key_is_rejected(self):
        for value in ("", "   "):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "API key is required"):
                    self.store.set_api_key(_IDENTITY, "openai", value)

    def test_key_is_encrypted_at_rest_and_round_trips(self):
        self.store.set_api_key(_IDENTITY, "OPENAI", "  sk-user-secret  ")
        with open(self.user_settings_file, encoding="utf-8") as handle:
            raw = handle.read()
        self.assertNotIn("sk-user-secret", raw)
        self.assertEqual(self.store.get_api_key(_IDENTITY, "openai"), "sk-user-secret")
        self.assertTrue(self.store.has_api_key(_IDENTITY, "openai"))

    def test_set_api_key_creates_the_other_record_sections(self):
        self.store.set_api_key(_IDENTITY, "openai", "sk-user")
        with open(self.user_settings_file, encoding="utf-8") as handle:
            raw = json.load(handle)["users"][_IDENTITY]
        self.assertEqual(raw["google_oauth"], {})
        self.assertEqual(raw["preferences"], self.store.default_preferences())

    def test_key_is_written_under_the_normalised_provider(self):
        self.store.set_api_key(_IDENTITY, "  OpenAI ", "sk-user")
        self.assertEqual(self.store.get_api_key(_IDENTITY, "OPENAI"), "sk-user")

    def test_unsupported_provider_or_identity_reads_as_none(self):
        self.store.set_api_key(_IDENTITY, "openai", "sk-user")
        self.assertIsNone(self.store.get_api_key(_IDENTITY, "ollama"))
        self.assertIsNone(self.store.get_api_key("", "openai"))
        self.assertFalse(self.store.has_api_key(_IDENTITY, "ollama"))

    def test_undecryptable_ciphertext_reads_as_none(self):
        self._write_record({"credentials": {"openai": "garbage-token"}})
        self.assertIsNone(self.store.get_api_key(_IDENTITY, "openai"))

    def test_non_string_ciphertext_reads_as_none(self):
        for value in (None, 123, {"nope": True}, ["nope"]):
            with self.subTest(value=value):
                self._write_record({"credentials": {"openai": value}})
                self.assertIsNone(self.store.get_api_key(_IDENTITY, "openai"))

    def test_empty_ciphertext_decrypts_to_none(self):
        self._write_record({"credentials": {"openai": ""}})
        self.assertIsNone(self.store.get_api_key(_IDENTITY, "openai"))

    def test_delete_removes_the_key(self):
        self.store.set_api_key(_IDENTITY, "openai", "sk-user")
        self.store.delete_api_key(_IDENTITY, "openai")
        self.assertIsNone(self.store.get_api_key(_IDENTITY, "openai"))

    def test_delete_of_unsupported_provider_or_identity_is_a_no_op(self):
        self.store.set_api_key(_IDENTITY, "openai", "sk-user")
        self.store.delete_api_key(_IDENTITY, "ollama")
        self.store.delete_api_key("", "openai")
        self.assertEqual(self.store.get_api_key(_IDENTITY, "openai"), "sk-user")

    def test_delete_on_a_record_without_credentials_is_a_no_op(self):
        self._write_record({"preferences": self.store.default_preferences()})
        self.store.delete_api_key(_IDENTITY, "openai")

    def test_delete_on_a_non_dict_record_is_a_no_op(self):
        self._write_record("not-a-dict")
        self.store.delete_api_key(_IDENTITY, "openai")

    def test_delete_on_an_absent_store_leaves_no_file_behind(self):
        self.store.delete_api_key(_IDENTITY, "openai")
        self.assertFalse(os.path.exists(self.user_settings_file))


class StoreFileLoadingTests(_StoreFixture):
    def test_missing_file_yields_an_empty_store(self):
        self.assertEqual(self.store._load(), {"users": {}})

    def test_unreadable_file_yields_an_empty_store(self):
        self._write_raw("{not json")
        store = UserSettingsStore(
            llm_service_access=self.access,
            llm_assignments_store=self.assignments,
        )
        self.assertEqual(store._load(), {"users": {}})

    def test_structurally_wrong_payload_yields_an_empty_store(self):
        for payload in ('{"users": []}', '{"users": "nope"}', "[1, 2, 3]"):
            with self.subTest(payload=payload):
                self._write_raw(payload)
                store = UserSettingsStore(
                    llm_service_access=self.access,
                    llm_assignments_store=self.assignments,
                )
                self.assertEqual(store._load(), {"users": {}})

    def test_stat_failure_yields_an_empty_store(self):
        # The file exists but cannot be stat'd (permissions, transient fs error):
        # the store must degrade to an empty payload instead of raising.
        self._write_raw({"users": {}})
        with self._store_file_stat_fails():
            self.assertEqual(self.store._load(), {"users": {}})

    def test_cache_is_reused_while_the_file_is_unchanged(self):
        self.store.set_api_key(_IDENTITY, "openai", "sk-user")
        self.store._cache_data = None
        self.store._cache_mtime_ns = None
        self.store._cache_size = None

        with patch("app.services.user_settings_store.json.load", wraps=json.load) as mocked:
            self.assertEqual(self.store.get_api_key(_IDENTITY, "openai"), "sk-user")
            self.assertEqual(self.store.get_api_key(_IDENTITY, "openai"), "sk-user")
        self.assertEqual(mocked.call_count, 1)

    def test_cache_is_invalidated_when_the_file_changes(self):
        self.store.set_api_key(_IDENTITY, "openai", "sk-user")
        self.store._cache_data = None
        self.store._cache_mtime_ns = None
        self.store._cache_size = None

        with open(self.user_settings_file, encoding="utf-8") as handle:
            data = json.load(handle)
        data["users"]["google:second@example.com"] = {
            "preferences": self.store.default_preferences(),
            "credentials": {},
            "google_oauth": {},
        }
        with open(self.user_settings_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        _, has_saved = self.store.get_user_state("google:second@example.com")
        self.assertTrue(has_saved)

    def test_save_updates_the_cache_in_place(self):
        self.store.set_api_key(_IDENTITY, "openai", "sk-user")
        self.assertIsNotNone(self.store._cache_mtime_ns)
        self.assertIsNotNone(self.store._cache_size)

    def test_save_survives_a_stat_failure_on_the_written_file(self):
        with self._store_file_stat_fails():
            self.store.save_preferences(_IDENTITY, {"provider": "openai", "model": "gpt-4o"})
        self.assertIsNone(self.store._cache_mtime_ns)
        self.assertIsNone(self.store._cache_size)
        with open(self.user_settings_file, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["users"][_IDENTITY]["preferences"]["model"], "gpt-4o")

    def test_save_failure_cleans_up_the_temp_file(self):
        with patch("app.services.user_settings_store.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.set_api_key(_IDENTITY, "openai", "sk-user")
        leftovers = [n for n in os.listdir(self.tmpdir.name) if n.startswith("user-settings-")]
        self.assertEqual(leftovers, [])


class GoogleOAuthStorageTests(_StoreFixture):
    def test_missing_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing identity"):
            self.store.set_google_oauth_tokens("", access_token="a", refresh_token="r", expires_in=60)

    def test_blank_access_token_is_rejected(self):
        for value in ("", "   "):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Missing access token"):
                    self.store.set_google_oauth_tokens(
                        _IDENTITY,
                        access_token=value,
                        refresh_token="r",
                        expires_in=60,
                    )

    def test_tokens_are_stored_encrypted_with_an_expiry(self):
        self.store.set_google_oauth_tokens(
            _IDENTITY,
            access_token="  access-token  ",
            refresh_token="  refresh-token  ",
            expires_in=3600,
        )
        with open(self.user_settings_file, encoding="utf-8") as handle:
            raw = handle.read()
        self.assertNotIn("access-token", raw)
        self.assertNotIn("refresh-token", raw)

        with open(self.user_settings_file, encoding="utf-8") as handle:
            record = json.load(handle)["users"][_IDENTITY]["google_oauth"]
        self.assertIn("expires_at", record)
        self.assertEqual(self.store._decrypt(record["access_token"]), "access-token")
        self.assertEqual(self.store._decrypt(record["refresh_token"]), "refresh-token")

    def test_blank_refresh_token_leaves_the_previous_one_untouched(self):
        self.store.set_google_oauth_tokens(
            _IDENTITY,
            access_token="a1",
            refresh_token="r1",
            expires_in=3600,
        )
        for refresh in (None, "", "   "):
            with self.subTest(refresh=refresh):
                self.store.set_google_oauth_tokens(
                    _IDENTITY,
                    access_token="a2",
                    refresh_token=refresh,
                    expires_in=3600,
                )
                record = self._oauth_record()
                self.assertEqual(self.store._decrypt(record["access_token"]), "a2")
                self.assertEqual(self.store._decrypt(record["refresh_token"]), "r1")

    def test_set_tokens_creates_the_other_record_sections(self):
        self.store.set_google_oauth_tokens(
            _IDENTITY,
            access_token="a",
            refresh_token="r",
            expires_in=3600,
        )
        with open(self.user_settings_file, encoding="utf-8") as handle:
            record = json.load(handle)["users"][_IDENTITY]
        self.assertEqual(record["credentials"], {})
        self.assertEqual(record["preferences"], self.store.default_preferences())

    def test_status_reflects_stored_tokens(self):
        self.assertFalse(self.store.get_google_oauth_status(_IDENTITY))
        self.store.set_google_oauth_tokens(_IDENTITY, access_token="a", refresh_token=None, expires_in=60)
        self.assertTrue(self.store.get_google_oauth_status(_IDENTITY))
        self.store.clear_google_oauth(_IDENTITY)
        self.assertFalse(self.store.get_google_oauth_status(_IDENTITY))

    def test_status_of_empty_identity_is_false(self):
        for identity in ("", "   "):
            with self.subTest(identity=identity):
                self.assertFalse(self.store.get_google_oauth_status(identity))

    def test_status_is_true_with_only_a_refresh_token(self):
        self._write_record(
            {"google_oauth": {"refresh_token": self.store._encrypt("r")}},
        )
        self.assertTrue(self.store.get_google_oauth_status(_IDENTITY))

    def test_status_is_false_for_a_non_dict_record_or_oauth_section(self):
        for record in ("not-a-dict", {"google_oauth": "nope"}, {"google_oauth": {}}):
            with self.subTest(record=record):
                self._write_record(record)
                self.assertFalse(self.store.get_google_oauth_status(_IDENTITY))

    def test_clear_removes_the_oauth_section(self):
        self.store.set_google_oauth_tokens(_IDENTITY, access_token="a", refresh_token="r", expires_in=60)
        self.store.clear_google_oauth(_IDENTITY)
        with open(self.user_settings_file, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["users"][_IDENTITY]["google_oauth"], {})

    def test_clear_of_empty_identity_is_a_no_op(self):
        self.store.clear_google_oauth("")
        self.assertFalse(os.path.exists(self.user_settings_file))

    def test_clear_on_a_non_dict_record_is_a_no_op(self):
        self._write_record("not-a-dict")
        self.store.clear_google_oauth(_IDENTITY)


class GoogleAccessTokenTests(_StoreFixture):
    def _write_oauth(self, *, access=None, refresh=None, expires_at=None):
        record = {}
        if access is not None:
            record["access_token"] = self.store._encrypt(access)
        if refresh is not None:
            record["refresh_token"] = self.store._encrypt(refresh)
        if expires_at is not None:
            record["expires_at"] = expires_at
        self._write_record({"google_oauth": record})

    def _get(self, payload=None, error=None, identity=_IDENTITY):
        client = _FakeOAuthClient(payload=payload, error=error)
        with patch(
            "app.services.user_settings_store.get_oauth_http_client",
            return_value=client,
        ):
            token = asyncio.run(self.store.get_google_access_token(identity))
        return token, client

    def test_empty_identity_returns_none(self):
        token, client = self._get(identity="")
        self.assertIsNone(token)
        self.assertEqual(client.calls, [])

    def test_no_tokens_at_all_returns_none_without_an_http_call(self):
        self._write_record({"google_oauth": {}})
        token, client = self._get()
        self.assertIsNone(token)
        self.assertEqual(client.calls, [])

    def test_undecryptable_access_token_is_treated_as_missing(self):
        self._write_record({"google_oauth": {"access_token": "garbage", "refresh_token": "garbage"}})
        token, client = self._get()
        self.assertIsNone(token)
        self.assertEqual(client.calls, [])

    def test_fresh_access_token_is_returned_without_refreshing(self):
        self._write_oauth(access="a1", refresh="r1", expires_at="2999-01-01T00:00:00+00:00")
        token, client = self._get()
        self.assertEqual(token, "a1")
        self.assertEqual(client.calls, [])

    def test_access_token_without_expiry_is_returned_as_is(self):
        self._write_oauth(access="a1")
        token, client = self._get()
        self.assertEqual(token, "a1")
        self.assertEqual(client.calls, [])

    def test_malformed_expiry_is_ignored_and_the_token_is_returned(self):
        self._write_oauth(access="a1", refresh="r1", expires_at="not-a-timestamp")
        token, client = self._get()
        self.assertEqual(token, "a1")
        self.assertEqual(client.calls, [])

    def test_naive_expiry_is_interpreted_as_utc(self):
        self._write_oauth(access="a1", refresh="r1", expires_at="2999-01-01T00:00:00")
        token, client = self._get()
        self.assertEqual(token, "a1")
        self.assertEqual(client.calls, [])

    def test_near_expiry_token_is_refreshed(self):
        soon = (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat()
        self._write_oauth(access="stale", refresh="r1", expires_at=soon)

        token, client = self._get({"access_token": "fresh", "expires_in": 3600})
        self.assertEqual(token, "fresh")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["url"], "https://oauth2.googleapis.com/token")
        self.assertEqual(client.calls[0]["data"]["refresh_token"], "r1")
        self.assertEqual(client.calls[0]["data"]["grant_type"], "refresh_token")
        self.assertEqual(client.calls[0]["data"]["client_id"], "cid")

        record = self._oauth_record()
        self.assertEqual(self.store._decrypt(record["access_token"]), "fresh")

    def test_refresh_persists_a_new_refresh_token_when_returned(self):
        self._write_oauth(access="stale", refresh="r1", expires_at="2000-01-01T00:00:00+00:00")
        token, _ = self._get(
            {"access_token": "fresh", "refresh_token": "r2", "expires_in": 120},
        )
        self.assertEqual(token, "fresh")
        record = self._oauth_record()
        self.assertEqual(self.store._decrypt(record["refresh_token"]), "r2")

    def test_refresh_without_a_new_refresh_token_keeps_the_old_one(self):
        self._write_oauth(access="stale", refresh="r1", expires_at="2000-01-01T00:00:00+00:00")
        self._get({"access_token": "fresh", "expires_in": 0})
        record = self._oauth_record()
        self.assertEqual(self.store._decrypt(record["refresh_token"]), "r1")

    def test_refresh_response_without_an_access_token_returns_none(self):
        self._write_oauth(access="stale", refresh="r1", expires_at="2000-01-01T00:00:00+00:00")
        token, _ = self._get({"expires_in": 3600})
        self.assertIsNone(token)

    def test_http_failure_returns_none(self):
        self._write_oauth(access="stale", refresh="r1", expires_at="2000-01-01T00:00:00+00:00")
        token, client = self._get(error=RuntimeError("network down"))
        self.assertIsNone(token)
        self.assertEqual(len(client.calls), 1)

    def test_refresh_uses_the_refresh_token_when_the_access_token_is_absent(self):
        self._write_oauth(refresh="r1", expires_at="2000-01-01T00:00:00+00:00")
        token, client = self._get({"access_token": "fresh", "expires_in": 3600})
        self.assertEqual(token, "fresh")
        self.assertEqual(client.calls[0]["data"]["refresh_token"], "r1")


class ResolvePersonalCredentialTests(_StoreFixture):
    def _resolve(self, provider, auth_mode="api_key", identity=_IDENTITY):
        return asyncio.run(
            self.store.resolve_personal_runtime_credential(
                identity_key=identity,
                provider=provider,
                auth_mode=auth_mode,
            )
        )

    def test_unsupported_provider_requires_a_personal_credential(self):
        self.assertEqual(
            self._resolve("ollama"),
            (None, "missing_personal", PERSONAL_CREDENTIAL_REQUIRED_CODE),
        )

    def test_api_key_mode_returns_the_stored_key(self):
        self.store.set_api_key(_IDENTITY, "openai", "sk-user")
        self.assertEqual(self._resolve("openai"), ("sk-user", "user_api_key", None))

    def test_missing_api_key_is_blocked(self):
        self.assertEqual(
            self._resolve("openai"),
            (None, "missing_personal", PERSONAL_CREDENTIAL_REQUIRED_CODE),
        )

    def test_provider_is_matched_case_insensitively(self):
        self.store.set_api_key(_IDENTITY, "openai", "sk-user")
        self.assertEqual(self._resolve("  OpenAI  ")[0], "sk-user")

    def test_google_account_mode_returns_a_bearer_dict(self):
        self.store.set_google_oauth_tokens(_IDENTITY, access_token="tok", refresh_token=None, expires_in=3600)
        self.assertEqual(
            self._resolve("google", auth_mode="account"),
            ({"Authorization": "Bearer tok"}, "user_account", None),
        )

    def test_google_api_key_mode_with_no_key_falls_back_to_the_account_token(self):
        self.store.set_google_oauth_tokens(_IDENTITY, access_token="tok", refresh_token=None, expires_in=3600)
        self.assertEqual(
            self._resolve("google", auth_mode="api_key"),
            ({"Authorization": "Bearer tok"}, "user_account", None),
        )

    def test_google_account_mode_prefers_the_account_token_over_an_api_key(self):
        self.store.set_google_oauth_tokens(_IDENTITY, access_token="tok", refresh_token=None, expires_in=3600)
        self.store.set_api_key(_IDENTITY, "google", "sk-google")
        self.assertEqual(
            self._resolve("google", auth_mode="account"),
            ({"Authorization": "Bearer tok"}, "user_account", None),
        )

    def test_google_api_key_mode_prefers_the_api_key_over_the_account_token(self):
        self.store.set_google_oauth_tokens(_IDENTITY, access_token="tok", refresh_token=None, expires_in=3600)
        self.store.set_api_key(_IDENTITY, "google", "sk-google")
        self.assertEqual(self._resolve("google", auth_mode="api_key"), ("sk-google", "user_api_key", None))

    def test_google_account_mode_without_a_token_is_blocked(self):
        self.assertEqual(
            self._resolve("google", auth_mode="account"),
            (None, "missing_personal", PERSONAL_CREDENTIAL_REQUIRED_CODE),
        )

    def test_empty_identity_is_blocked(self):
        self.assertEqual(
            self._resolve("openai", identity=""),
            (None, "missing_personal", PERSONAL_CREDENTIAL_REQUIRED_CODE),
        )


class StudyAppResolutionTests(_StoreFixture):
    def test_unapproved_user_is_blocked(self):
        self.assertEqual(
            self.store.resolve_study_app_provider_model(identity_key=_IDENTITY),
            (None, None, APPROVAL_REQUIRED_CODE),
        )

    def test_approved_user_without_an_assignment_is_blocked(self):
        self.access.add_user("google", "learner@example.com")
        self.assertEqual(
            self.store.resolve_study_app_provider_model(identity_key=_IDENTITY),
            (None, None, STUDY_APP_NOT_ASSIGNED_CODE),
        )

    def test_approved_user_with_an_assignment_gets_provider_and_model(self):
        self.access.add_user("google", "learner@example.com")
        self.assignments.set_assignment(
            identity_key=_IDENTITY,
            provider="openai",
            model="gpt-4o-mini",
        )
        self.assertEqual(
            self.store.resolve_study_app_provider_model(identity_key=_IDENTITY),
            ("openai", "gpt-4o-mini", None),
        )

    def test_bare_identifier_matches_a_provider_scoped_allowlist_entry(self):
        self.access.add_user("", "learner@example.com")
        self.assertEqual(
            self.store.resolve_study_app_provider_model(identity_key="learner@example.com")[2],
            STUDY_APP_NOT_ASSIGNED_CODE,
        )

    def test_backend_fallback_approval_check(self):
        self.assertFalse(self.store.is_identity_approved_for_backend_fallback(_IDENTITY))
        self.access.add_user("google", "learner@example.com")
        self.assertTrue(self.store.is_identity_approved_for_backend_fallback(_IDENTITY))

    def test_assignment_read_returns_none_when_unset(self):
        self.assertIsNone(self.store.get_study_app_assignment(_IDENTITY))

    def test_assignment_read_exposes_provider_model_and_timestamp(self):
        created = self.assignments.set_assignment(
            identity_key=_IDENTITY,
            provider="openai",
            model="gpt-4o-mini",
        )
        self.assertEqual(
            self.store.get_study_app_assignment(_IDENTITY),
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "updated_at": created.updated_at,
            },
        )

    def test_provider_models_are_delegated_to_the_assignments_store(self):
        self.assertEqual(
            self.store.get_study_app_provider_models(),
            self.assignments.provider_models(),
        )

    def test_identity_key_is_normalised_by_the_assignments_store(self):
        self.assertEqual(self.store.get_identity_key(" Google ", " A@X.com "), "google:a@x.com")
        self.assertEqual(self.store.get_identity_key("", "a@x.com"), "")


if __name__ == "__main__":
    unittest.main()