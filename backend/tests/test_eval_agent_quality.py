import unittest
from pathlib import Path

from scripts.eval_agent_quality import run_eval


class EvalAgentQualityTests(unittest.TestCase):
    def test_eval_harness_outputs_expected_metrics(self):
        fixture_dir = Path(__file__).resolve().parents[1] / "evals"
        report = run_eval(fixture_dir)
        summary = report["summary"]
        self.assertGreaterEqual(summary["schema_valid_rate"], 98.0)
        self.assertGreaterEqual(summary["grounding_fidelity"], 50.0)
        self.assertGreaterEqual(report["flows"]["questions"]["total"], 1)


if __name__ == "__main__":
    unittest.main()
