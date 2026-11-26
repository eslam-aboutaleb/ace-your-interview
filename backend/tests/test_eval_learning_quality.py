"""Tests for the learning-quality eval harness (issue #17).

Asserts the harness binds production code (never local copies), the
fixtures stay parseable, the deterministic gate passes on the checked-in
fixtures, and the gate actually fails when a metric regresses.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = BACKEND_ROOT / "scripts" / "eval_learning_quality.py"

spec = importlib.util.spec_from_file_location("eval_learning_quality", SCRIPT)
harness = importlib.util.module_from_spec(spec)
sys.modules.setdefault("eval_learning_quality", harness)
spec.loader.exec_module(harness)


class ProductionBindingTests(unittest.TestCase):
    """A local copy of a validator cannot be substituted silently."""

    def test_harness_uses_production_validators(self):
        from app.services import learning_store, question_generator

        self.assertIs(harness.LearningStore, learning_store.LearningStore)
        self.assertIs(
            harness._is_near_duplicate_question,
            question_generator._is_near_duplicate_question,
        )
        self.assertIs(
            harness._validate_question_item,
            question_generator._validate_question_item,
        )
        self.assertIs(
            harness._normalise_question, question_generator._normalise_question
        )

    def test_harness_calls_no_model(self):
        import inspect

        source = inspect.getsource(harness)
        self.assertNotIn("LLMClient", source)
        self.assertNotIn("completion(", source)


class FixtureTests(unittest.TestCase):
    def test_semantic_duplicate_fixture_is_balanced_and_labeled(self):
        pairs = harness._load_jsonl(harness.EVALS_DIR / "semantic_duplicates.jsonl")
        self.assertGreaterEqual(len(pairs), 6)
        labels = {p["label"] for p in pairs}
        self.assertEqual(labels, {"duplicate", "not_duplicate"})
        for pair in pairs:
            self.assertTrue(pair["a"].strip())
            self.assertTrue(pair["b"].strip())
            self.assertTrue(pair["rationale"].strip())

    def test_journey_sessions_fixture_shape(self):
        sessions = harness._load_jsonl(harness.EVALS_DIR / "journey_sessions.jsonl")
        self.assertGreaterEqual(len(sessions), 2)
        for session in sessions:
            self.assertTrue(session["steps"])
            expected = session["expect"]
            self.assertEqual(expected["attempts_total"], len(session["steps"]))


class GateTests(unittest.TestCase):
    def test_checked_in_fixtures_pass_the_gate(self):
        report = harness.run_eval()
        self.assertTrue(report["passed"], report["gate_failures"])
        self.assertEqual(report["metrics"]["exact_duplicate_count"], 0)
        self.assertEqual(report["metrics"]["schema_valid_rate"], 100.0)
        self.assertEqual(report["metrics"]["progress_replay_match_rate"], 100.0)

    def test_report_records_versions(self):
        report = harness.run_eval()
        self.assertEqual(report["version"]["harness_tier"], "deterministic")
        self.assertTrue(report["version"]["prompt_snapshot_cases"])

    def test_gate_fails_on_schema_regression(self):
        metrics = {
            "schema_valid_rate": 80.0,
            "exact_duplicate_count": 0,
            "duplicate_precision": 100.0,
            "duplicate_recall": 100.0,
            "progress_replay_match_rate": 100.0,
            "objective_coverage": 100.0,
        }
        failures = harness.evaluate_gates(metrics)
        self.assertEqual([f["gate"] for f in failures], ["schema_valid_rate"])

    def test_gate_fails_on_progress_loss_and_duplicates(self):
        metrics = {
            "schema_valid_rate": 100.0,
            "exact_duplicate_count": 1,
            "duplicate_precision": 100.0,
            "duplicate_recall": 100.0,
            "progress_replay_match_rate": 50.0,
            "objective_coverage": 100.0,
        }
        gates = {f["gate"] for f in harness.evaluate_gates(metrics)}
        self.assertEqual(gates, {"exact_duplicate_count", "progress_replay_match_rate"})


if __name__ == "__main__":
    unittest.main()
