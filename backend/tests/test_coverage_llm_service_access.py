"""Coverage for ``app.services.llm_service_access``.

The allowlist decides who may spend the *backend's* LLM budget, so the tests
focus on the trust boundary: seeding from ``STUDY_ADMIN_USERS``, corrupt-file
recovery, normalisation of provider-scoped vs bare entries, and removal of both
spellings.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from app.config import get_settings
from app.services.llm_service_access import (
    LLMServiceAccess,
    _normalise_admin_entries,
    _normalise_entry,
)


class NormalisationTests(unittest.TestCase):
    def test_entry_is_lowercased_and_provider_scoped(self):
        self.assertEqual(_normalise_entry("  Google ", " Learner@Example.COM "), "google:learner@example.com")

    def test_entry_without_provider_is_the_bare_identifier(self):
        self.assertEqual(_normalise_entry("", "  Learner@Example.COM "), "learner@example.com")
        self.assertEqual(_normalise_entry(None, "learner@example.com"), "learner@example.com")

    def test_entry_without_identifier_is_empty(self):
        for identifier in ("", "   ", None):
            with self.subTest(identifier=identifier):
                self.assertEqual(_normalise_entry("google", identifier), "")

    def test_admin_entries_drop_blanks_and_duplicates(self):
        parsed = _normalise_admin_entries(" Alice , ,BOB,alice, Bob ,carol ")
        self.assertEqual(parsed, ["alice", "bob", "carol"])

    def test_admin_entries_of_empty_input(self):
        self.assertEqual(_normalise_admin_entries(""), [])
        self.assertEqual(_normalise_admin_entries(None), [])
        self.assertEqual(_normalise_admin_entries(" , , "), [])


class _AccessFixture(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.users_file = os.path.join(self.tmpdir.name, "llm_service_users.json")
        self.prev_env = {
            "STUDY_LLM_SERVICE_USERS_FILE": os.environ.get("STUDY_LLM_SERVICE_USERS_FILE"),
            "STUDY_ADMIN_USERS": os.environ.get("STUDY_ADMIN_USERS"),
        }
        os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = self.users_file
        os.environ["STUDY_ADMIN_USERS"] = ""
        get_settings.cache_clear()

    def tearDown(self):
        self.tmpdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _write(self, payload):
        with open(self.users_file, "w", encoding="utf-8") as handle:
            if isinstance(payload, str):
                handle.write(payload)
            else:
                json.dump(payload, handle)


class SeedingAndLoadingTests(_AccessFixture):
    def test_first_access_seeds_from_admin_users(self):
        os.environ["STUDY_ADMIN_USERS"] = " Root@Example.com , other , root@example.com "
        get_settings.cache_clear()

        access = LLMServiceAccess()
        self.assertEqual(access.list_users(), {"users": ["root@example.com", "other"]})
        with open(self.users_file, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"users": ["root@example.com", "other"]})

    def test_corrupt_file_falls_back_to_seeding_from_admin_users(self):
        self._write("{not json")
        os.environ["STUDY_ADMIN_USERS"] = "seeded@example.com"
        get_settings.cache_clear()

        access = LLMServiceAccess()
        self.assertEqual(access.list_users(), {"users": ["seeded@example.com"]})

    def test_file_with_non_list_users_is_replaced_by_seed(self):
        self._write({"users": {"a": 1}})
        os.environ["STUDY_ADMIN_USERS"] = "seeded@example.com"
        get_settings.cache_clear()

        access = LLMServiceAccess()
        self.assertEqual(access.list_users(), {"users": ["seeded@example.com"]})

    def test_existing_file_entries_are_normalised_and_deduped(self):
        self._write({"users": ["  Alice ", "ALICE", "", None, "bob", 42]})
        os.environ["STUDY_ADMIN_USERS"] = "seeded@example.com"
        get_settings.cache_clear()

        access = LLMServiceAccess()
        # Blanks are dropped, casing/spacing is normalised and repeats collapse.
        # Non-string entries are dropped outright rather than stringified:
        # `str(None)` would admit a literal "none" identity into the allowlist,
        # and `str(42)` would invent a user named "42".
        self.assertEqual(access.list_users(), {"users": ["alice", "bob"]})

    def test_non_string_entries_never_become_identities(self):
        self._write({"users": [None, True, 42, 1.5, [], {}, "real-user"]})
        os.environ["STUDY_ADMIN_USERS"] = ""
        get_settings.cache_clear()

        self.assertEqual(LLMServiceAccess().list_users(), {"users": ["real-user"]})

    def test_missing_file_with_no_admins_yields_empty_allowlist(self):
        access = LLMServiceAccess()
        self.assertEqual(access.list_users(), {"users": []})
        self.assertTrue(os.path.exists(self.users_file))


class MutationTests(_AccessFixture):
    def test_add_user_appends_and_sorts(self):
        access = LLMServiceAccess()
        self.assertEqual(access.add_user("google", "zoe@example.com")["users"], ["google:zoe@example.com"])
        self.assertEqual(
            access.add_user("github", "amy@example.com")["users"],
            ["github:amy@example.com", "google:zoe@example.com"],
        )

    def test_add_user_is_idempotent_and_does_not_rewrite_the_file(self):
        access = LLMServiceAccess()
        access.add_user("google", "amy@example.com")
        with open(self.users_file, encoding="utf-8") as handle:
            first = json.load(handle)

        self.assertEqual(access.add_user("GOOGLE", "amy@example.com")["users"], ["google:amy@example.com"])
        with open(self.users_file, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), first)

    def test_add_user_without_identifier_is_a_no_op(self):
        access = LLMServiceAccess()
        access.add_user("google", "amy@example.com")
        self.assertEqual(access.add_user("google", "   ")["users"], ["google:amy@example.com"])
        self.assertEqual(access.add_user("google", None)["users"], ["google:amy@example.com"])

    def test_remove_user_drops_both_scoped_and_bare_spellings(self):
        access = LLMServiceAccess()
        access.add_user("google", "amy@example.com")
        access.add_user("", "amy@example.com")
        access.add_user("google", "bob@example.com")
        self.assertEqual(
            sorted(access.list_users()["users"]),
            ["amy@example.com", "google:amy@example.com", "google:bob@example.com"],
        )

        remaining = access.remove_user("google", "amy@example.com")["users"]
        self.assertEqual(remaining, ["google:bob@example.com"])

    def test_remove_user_without_identifier_is_a_no_op(self):
        access = LLMServiceAccess()
        access.add_user("google", "amy@example.com")
        self.assertEqual(access.remove_user("google", "")["users"], ["google:amy@example.com"])

    def test_remove_unknown_user_is_a_no_op(self):
        access = LLMServiceAccess()
        access.add_user("google", "amy@example.com")
        self.assertEqual(access.remove_user("github", "nobody@example.com")["users"], ["google:amy@example.com"])

    def test_save_failure_cleans_up_the_temp_file(self):
        access = LLMServiceAccess()
        with patch("app.services.llm_service_access.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                access.add_user("google", "amy@example.com")

        leftovers = [n for n in os.listdir(self.tmpdir.name) if n.startswith("llm-service-users-")]
        self.assertEqual(leftovers, [])


class IsUserAllowedTests(_AccessFixture):
    def test_provider_scoped_entry_allows_only_its_provider(self):
        access = LLMServiceAccess()
        access.add_user("google", "amy@example.com")

        self.assertTrue(access.is_user_allowed(user="amy@example.com", provider="google"))
        self.assertFalse(access.is_user_allowed(user="amy@example.com", provider="github"))

    def test_bare_entry_allows_any_provider(self):
        access = LLMServiceAccess()
        access.add_user("", "amy@example.com")

        self.assertTrue(access.is_user_allowed(user="amy@example.com", provider="google"))
        self.assertTrue(access.is_user_allowed(user="amy@example.com", provider=""))
        self.assertTrue(access.is_user_allowed(user="amy@example.com", provider="anything"))

    def test_user_and_provider_are_matched_case_insensitively(self):
        access = LLMServiceAccess()
        access.add_user("Google", "Amy@Example.com")

        self.assertTrue(access.is_user_allowed(user="  AMY@EXAMPLE.COM ", provider="GOOGLE"))

    def test_empty_user_is_never_allowed(self):
        access = LLMServiceAccess()
        access.add_user("", "")
        access.add_user("google", "amy@example.com")
        for user in ("", "   ", None):
            with self.subTest(user=user):
                self.assertFalse(access.is_user_allowed(user=user, provider="google"))

    def test_unknown_user_is_not_allowed(self):
        access = LLMServiceAccess()
        access.add_user("google", "amy@example.com")
        self.assertFalse(access.is_user_allowed(user="bob@example.com", provider="google"))
        self.assertFalse(access.is_user_allowed(user="bob@example.com", provider=""))

    def test_allowlist_persists_across_store_instances(self):
        LLMServiceAccess().add_user("google", "amy@example.com")
        self.assertTrue(LLMServiceAccess().is_user_allowed(user="amy@example.com", provider="google"))


if __name__ == "__main__":
    unittest.main()