"""Coverage for ``app.services.llm_assignments_store``.

Admin-assigned provider/model pairs are what a user actually gets when they pick
"Study App LLM", so the tests pin the trust boundary hard: only whitelisted
providers, only models from that provider's catalogue, and reads that validate
again because the JSON file is admin-editable.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from app.config import get_settings
from app.services.llm_assignments_store import (
    LLMAssignmentsStore,
    normalise_identity,
)


class NormaliseIdentityTests(unittest.TestCase):
    def test_identity_is_lowercased_and_stripped(self):
        self.assertEqual(normalise_identity(" Google ", " Learner@Example.COM "), "google:learner@example.com")

    def test_missing_provider_or_identifier_yields_empty_string(self):
        self.assertEqual(normalise_identity("", "learner@example.com"), "")
        self.assertEqual(normalise_identity("google", ""), "")
        self.assertEqual(normalise_identity(None, None), "")


class _AssignmentsFixture(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.assignments_file = os.path.join(self.tmpdir.name, "llm_assignments.json")
        self.prev_env = {
            "STUDY_LLM_ASSIGNMENTS_FILE": os.environ.get("STUDY_LLM_ASSIGNMENTS_FILE"),
        }
        os.environ["STUDY_LLM_ASSIGNMENTS_FILE"] = self.assignments_file
        get_settings.cache_clear()
        self.store = LLMAssignmentsStore()

    def tearDown(self):
        self.tmpdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _write(self, payload):
        with open(self.assignments_file, "w", encoding="utf-8") as handle:
            if isinstance(payload, str):
                handle.write(payload)
            else:
                json.dump(payload, handle)


class SetAssignmentTests(_AssignmentsFixture):
    def test_assignment_is_stored_and_returned(self):
        assignment = self.store.set_assignment(
            identity_key="Google:Learner@Example.com",
            provider="OpenAI",
            model=" gpt-4o-mini ",
        )
        self.assertEqual(assignment.provider, "openai")
        self.assertEqual(assignment.model, "gpt-4o-mini")
        self.assertTrue(assignment.updated_at)

        with open(self.assignments_file, encoding="utf-8") as handle:
            raw = json.load(handle)
        self.assertIn("google:learner@example.com", raw["assignments"])

    def test_assignment_is_updated_in_place(self):
        self.store.set_assignment(identity_key="google:a@x.com", provider="openai", model="gpt-4o")
        self.store.set_assignment(identity_key="google:a@x.com", provider="openai", model="gpt-4o-mini")
        self.assertEqual(self.store.get_assignment(identity_key="google:a@x.com").model, "gpt-4o-mini")

    def test_missing_identity_is_rejected(self):
        for identity in ("", "   ", None):
            with self.subTest(identity=identity):
                with self.assertRaisesRegex(ValueError, "Missing identity"):
                    self.store.set_assignment(
                        identity_key=identity,
                        provider="openai",
                        model="gpt-4o",
                    )

    def test_unsupported_provider_is_rejected(self):
        for provider in ("ollama", "cohere", "", None):
            with self.subTest(provider=provider):
                with self.assertRaisesRegex(ValueError, "Unsupported provider"):
                    self.store.set_assignment(
                        identity_key="google:a@x.com",
                        provider=provider,
                        model="gpt-4o",
                    )

    def test_unsupported_or_empty_model_is_rejected(self):
        for model in ("gpt-5-turbo", "", "   ", None):
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "Unsupported model"):
                    self.store.set_assignment(
                        identity_key="google:a@x.com",
                        provider="openai",
                        model=model,
                    )

    def test_model_from_a_different_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model"):
            self.store.set_assignment(
                identity_key="google:a@x.com",
                provider="openai",
                model="claude-3-5-sonnet-20241022",
            )

    def test_save_failure_cleans_up_the_temp_file(self):
        with patch("app.services.llm_assignments_store.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.set_assignment(
                    identity_key="google:a@x.com",
                    provider="openai",
                    model="gpt-4o",
                )

        leftovers = [n for n in os.listdir(self.tmpdir.name) if n.startswith("llm-assignments-")]
        self.assertEqual(leftovers, [])


class GetAssignmentTests(_AssignmentsFixture):
    def test_missing_file_returns_none(self):
        self.assertIsNone(self.store.get_assignment(identity_key="google:a@x.com"))

    def test_empty_identity_returns_none(self):
        for identity in ("", "   ", None):
            with self.subTest(identity=identity):
                self.assertIsNone(self.store.get_assignment(identity_key=identity))

    def test_corrupt_file_returns_none(self):
        self._write("{not json")
        self.assertIsNone(self.store.get_assignment(identity_key="google:a@x.com"))

    def test_file_with_non_dict_assignments_returns_none(self):
        self._write({"assignments": []})
        self.assertIsNone(self.store.get_assignment(identity_key="google:a@x.com"))

    def test_non_dict_record_returns_none(self):
        self._write({"assignments": {"google:a@x.com": "not-a-dict"}})
        self.assertIsNone(self.store.get_assignment(identity_key="google:a@x.com"))

    def test_record_with_unwhitelisted_provider_returns_none(self):
        self._write(
            {"assignments": {"google:a@x.com": {"provider": "ollama", "model": "llama3.2", "updated_at": "t"}}}
        )
        self.assertIsNone(self.store.get_assignment(identity_key="google:a@x.com"))

    def test_record_with_empty_model_returns_none(self):
        self._write({"assignments": {"google:a@x.com": {"provider": "openai", "model": "  ", "updated_at": "t"}}})
        self.assertIsNone(self.store.get_assignment(identity_key="google:a@x.com"))

    def test_record_with_model_outside_the_provider_catalogue_returns_none(self):
        # A hand-edited (or stale) file must not hand a user an arbitrary model.
        self._write(
            {"assignments": {"google:a@x.com": {"provider": "openai", "model": "gpt-9-imaginary", "updated_at": "t"}}}
        )
        self.assertIsNone(self.store.get_assignment(identity_key="google:a@x.com"))

    def test_provider_and_model_are_normalised_on_read(self):
        self._write(
            {
                "assignments": {
                    "google:a@x.com": {"provider": " OpenAI ", "model": " gpt-4o-mini ", "updated_at": " t "}
                }
            }
        )
        assignment = self.store.get_assignment(identity_key="google:a@x.com")
        self.assertEqual(assignment.provider, "openai")
        self.assertEqual(assignment.model, "gpt-4o-mini")
        self.assertEqual(assignment.updated_at, "t")

    def test_lookup_is_case_insensitive(self):
        self.store.set_assignment(identity_key="google:a@x.com", provider="openai", model="gpt-4o")
        self.assertEqual(self.store.get_assignment(identity_key="GOOGLE:A@X.COM").model, "gpt-4o")


class DeleteAssignmentTests(_AssignmentsFixture):
    def test_delete_reports_whether_the_assignment_existed(self):
        self.store.set_assignment(identity_key="google:a@x.com", provider="openai", model="gpt-4o")
        self.assertTrue(self.store.delete_assignment(identity_key="google:a@x.com"))
        self.assertIsNone(self.store.get_assignment(identity_key="google:a@x.com"))
        self.assertFalse(self.store.delete_assignment(identity_key="google:a@x.com"))

    def test_empty_identity_returns_false_without_writing(self):
        for identity in ("", "   ", None):
            with self.subTest(identity=identity):
                self.assertFalse(self.store.delete_assignment(identity_key=identity))
        self.assertFalse(os.path.exists(self.assignments_file))

    def test_delete_preserves_other_assignments(self):
        self.store.set_assignment(identity_key="google:a@x.com", provider="openai", model="gpt-4o")
        self.store.set_assignment(identity_key="google:b@x.com", provider="openai", model="gpt-4o-mini")
        self.store.delete_assignment(identity_key="google:a@x.com")
        self.assertEqual(
            sorted(self.store.list_assignments()),
            ["google:b@x.com"],
        )


class ListAssignmentsTests(_AssignmentsFixture):
    def test_empty_store_lists_nothing(self):
        self.assertEqual(self.store.list_assignments(), {})

    def test_list_skips_records_that_fail_validation(self):
        self._write(
            {
                "assignments": {
                    "google:good@x.com": {"provider": "openai", "model": "gpt-4o", "updated_at": "t"},
                    "google:bad-provider@x.com": {"provider": "ollama", "model": "llama3.2", "updated_at": "t"},
                    "google:bad-model@x.com": {"provider": "openai", "model": "nope", "updated_at": "t"},
                    "google:not-a-dict@x.com": "oops",
                }
            }
        )
        listed = self.store.list_assignments()
        self.assertEqual(list(listed), ["google:good@x.com"])
        self.assertEqual(listed["google:good@x.com"].model, "gpt-4o")

    def test_corrupt_file_lists_nothing(self):
        self._write("not json at all")
        self.assertEqual(self.store.list_assignments(), {})


class ProviderModelsTests(_AssignmentsFixture):
    def test_every_whitelisted_provider_gets_its_catalogue(self):
        models = self.store.provider_models()
        self.assertEqual(set(models), {"google", "openai", "anthropic", "groq"})
        self.assertIn("gpt-4o-mini", models["openai"])
        self.assertIn("gemini-2.5-pro", models["google"])


if __name__ == "__main__":
    unittest.main()