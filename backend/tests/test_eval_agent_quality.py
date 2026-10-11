"""The eval harness must share code with production, and its metrics must be able to fail."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts import eval_agent_quality as harness
from scripts.eval_agent_quality import (
    CALIBRATION_DIMENSIONS,
    CALIBRATION_FILE,
    DEFAULT_THRESHOLDS,
    FLOW_CONTRACTS,
    FORBIDDEN_FIXTURE_KEYS,
    PRODUCTION_VALIDATORS,
    compare_calibration,
    load_calibration_fixtures,
    load_cases,
    prompt_contract_violations,
    rebuild_prompt,
    retry_distribution,
    run_eval,
)

BACKEND = Path(__file__).resolve().parents[1]
EVALS = BACKEND / "evals"


class HarnessBindsProductionValidatorsTests(unittest.TestCase):
    """Assert the import in a test, not by inspection.

    Each case binds a harness name to the production object and checks identity
    against a fresh import from ``app.services``. Re-implementing any of these
    locally — which is the bug this harness exists to remove — fails here.
    """

    def test_question_flow_binds_question_generator_validators(self):
        from app.services.question_generator import (
            _parse_questions_json,
            _validate_question_item,
            _validate_quiz_item,
        )

        self.assertIs(PRODUCTION_VALIDATORS["validate_question_item"], _validate_question_item)
        self.assertIs(PRODUCTION_VALIDATORS["validate_quiz_item"], _validate_quiz_item)
        self.assertIs(PRODUCTION_VALIDATORS["parse_questions_json"], _parse_questions_json)

    def test_report_is_produced_by_calling_production_validators(self):
        """A copied validator would pass an identity check but fail this one.

        The harness module's global is replaced with a sentinel; if the report
        does not move, the harness is deciding validity somewhere else.
        """
        from app.services.question_generator import _validate_quiz_item

        original = harness._validate_quiz_item
        harness._validate_quiz_item = lambda *args, **kwargs: (False, "sentinel_rejection")
        try:
            sabotaged = run_eval(EVALS)
        finally:
            harness._validate_quiz_item = original
        self.assertLess(sabotaged["summary"]["schema_valid_rate"], 100.0)
        self.assertIn("schema_valid_rate", {f["gate"] for f in sabotaged["gate_failures"]})
        self.assertEqual(run_eval(EVALS)["summary"]["schema_valid_rate"], 100.0)
        self.assertEqual(_validate_quiz_item.__module__, "app.services.question_generator")

    def test_interview_flow_binds_interview_generator_validators(self):
        from app.services.interview_generator import InterviewGenerator

        self.assertIs(
            PRODUCTION_VALIDATORS["validate_interview_question_payload"],
            InterviewGenerator._validate_question_payload,
        )
        self.assertIs(
            PRODUCTION_VALIDATORS["validate_interview_eval_payload"],
            InterviewGenerator._validate_eval_payload,
        )
        self.assertIs(
            PRODUCTION_VALIDATORS["normalise_eval_payload"],
            InterviewGenerator._normalise_eval_payload,
        )
        self.assertIs(
            harness._INTERVIEW_EVAL_NORMALISER.__class__, InterviewGenerator
        )

    def test_shared_json_parser_and_question_normaliser_come_from_production(self):
        from app.services.llm_client import parse_json_object
        from app.services.question_text import normalise_question

        self.assertIs(PRODUCTION_VALIDATORS["parse_json_object"], parse_json_object)
        self.assertIs(PRODUCTION_VALIDATORS["normalise_question"], normalise_question)

    def test_prompt_blocks_come_from_production(self):
        from app.services.prompt_blocks import neutralise_fence, render_contract, untrusted_block

        self.assertIs(PRODUCTION_VALIDATORS["render_contract"], render_contract)
        self.assertIs(PRODUCTION_VALIDATORS["untrusted_block"], untrusted_block)
        self.assertIs(PRODUCTION_VALIDATORS["neutralise_fence"], neutralise_fence)

    def test_harness_defines_no_local_validator_shadowing_production(self):
        """The deleted functions must stay deleted."""
        for name in ("_validate_questions", "_validate_quiz", "_validate_interview", "_norm"):
            self.assertFalse(
                hasattr(harness, name),
                f"{name} is a local copy of production logic; import the production one instead",
            )

    def test_flow_contracts_are_derived_from_the_production_validators(self):
        """The required-field sets come from the validators, not from a hand list."""
        self.assertEqual(FLOW_CONTRACTS["questions"].fields, frozenset({"question", "answer"}))
        self.assertEqual(
            FLOW_CONTRACTS["interview"].fields,
            frozenset(
                {
                    "rubric",
                    "technical_accuracy",
                    "reasoning_depth",
                    "communication_clarity",
                    "completeness",
                    "confidence_signal",
                    "overall",
                    "strengths",
                    "improvements",
                    "follow_up_note",
                }
            ),
        )
        self.assertIn("competency_focus", FLOW_CONTRACTS["interview_question"].fields)
        self.assertIn("correct_answer", FLOW_CONTRACTS["quiz"].fields)


class HarnessMetricsCanFailTests(unittest.TestCase):
    def test_checked_in_fixtures_pass_every_deterministic_gate(self):
        report = run_eval(EVALS)
        self.assertEqual(report["gate_failures"], [])
        self.assertEqual(report["failing_cases"], [])
        self.assertEqual(report["summary"]["schema_valid_rate"], 100.0)
        self.assertEqual(report["summary"]["prompt_contract_rate"], 100.0)
        self.assertEqual(report["summary"]["prompt_output_field_rate"], 100.0)
        self.assertEqual(report["summary"]["untrusted_containment_rate"], 100.0)
        self.assertEqual(report["summary"]["prompt_digest_match_rate"], 100.0)
        self.assertEqual(report["summary"]["duplicate_rate"], 0.0)

    def test_fixture_matrix_covers_the_documented_dimensions(self):
        from app.services.topic_catalog import PROBLEM_SOLVING_LANGUAGE_OPTIONS

        cases = load_cases(EVALS)
        self.assertGreaterEqual(len(cases), 40)

        languages = set()
        levels = set()
        interview_types = set()
        for case in cases:
            levels.update(tag for tag in case.tags if tag in {"junior", "mid", "senior"})
            interview_types.update(
                tag for tag in case.tags if tag in {"coding", "behavioral", "mixed"}
            )
            languages.update(tag for tag in case.tags if tag in PROBLEM_SOLVING_LANGUAGE_OPTIONS)
        self.assertEqual(levels, {"junior", "mid", "senior"})
        self.assertEqual(interview_types, {"coding", "behavioral", "mixed"})
        self.assertEqual(languages, set(PROBLEM_SOLVING_LANGUAGE_OPTIONS))

    def test_matrix_coverage_gate_fails_when_a_language_is_added_without_a_fixture(self):
        from app.services.topic_catalog import PROBLEM_SOLVING_LANGUAGE_OPTIONS

        cases = load_cases(EVALS)
        coverage = harness.matrix_coverage(cases)
        self.assertEqual(coverage["problem_solving_language_missing"], [])
        self.assertEqual(coverage["level_missing"], [])
        self.assertEqual(coverage["interview_type_missing"], [])
        # The gate is derived from the production option list, so a new language
        # shows up as a missing fixture rather than silently going untested.
        self.assertEqual(
            coverage["problem_solving_language"], sorted(PROBLEM_SOLVING_LANGUAGE_OPTIONS)
        )
        self.assertEqual(harness._matrix_failures(coverage), [])

        narrowed = dict(coverage, problem_solving_language_missing=["kotlin"])
        self.assertEqual(
            [failure["gate"] for failure in harness._matrix_failures(narrowed)],
            ["matrix_problem_solving_language_coverage"],
        )

    def test_attempt_budget_fixtures_exist_and_they_agree_with_the_budget(self):
        from app.services.question_generator import _attempt_budget

        fan_out = [case for case in load_cases(EVALS) if "attempt_budget" in case.tags]
        self.assertGreaterEqual(len(fan_out), 3)
        self.assertTrue(any(case.item_count >= 6 for case in fan_out))
        # The counts the fan-out cases ask for must actually move the budget
        # above the base, otherwise the case is not exercising anything.
        counts = [
            json.loads(line)["input"]["count"]
            for name in ("questions.jsonl", "quiz.jsonl")
            for line in (EVALS / name).read_text(encoding="utf-8").splitlines()
            if line.strip() and "attempt_budget" in json.loads(line).get("tags", [])
        ]
        self.assertTrue(counts)
        self.assertTrue(any(_attempt_budget(count) > 5 for count in counts))

    def test_repair_rate_is_measured_from_the_production_repair_flag(self):
        report = run_eval(EVALS)
        eval_cases = [case for case in load_cases(EVALS) if case.flow == "interview"]
        repaired = [case for case in eval_cases if case.note_repaired]
        self.assertTrue(repaired, "no fixture exercises the follow_up_note repair")
        expected = round(len(repaired) / len(eval_cases) * 100, 2)
        self.assertEqual(report["summary"]["repair_rate"], expected)

    def test_degraded_eval_rate_is_measured_and_gated(self):
        report = run_eval(EVALS)
        eval_cases = [case for case in load_cases(EVALS) if case.flow == "interview"]
        degraded = [case for case in eval_cases if case.degraded]
        self.assertTrue(degraded, "no fixture exercises the degraded path")
        for case in degraded:
            self.assertFalse(case.valid)
            self.assertFalse(case.expected_valid)
        self.assertLessEqual(
            report["summary"]["degraded_eval_rate"], DEFAULT_THRESHOLDS["degraded_eval_rate_max"]
        )

    def test_repair_rate_gate_fails_when_the_prompt_regresses(self):
        with self._mutated_fixtures() as fixture_dir:
            self.assertEqual(_failure_gates(fixture_dir), set())

            # Empty rubrics: exactly what a model emits when the evaluate prompt
            # stops showing the rubric shape, and exactly what the degraded
            # path then flags. Both rates must move.
            self._rewrite(
                fixture_dir / "interview.jsonl",
                lambda row: row.update(
                    output=json.dumps({**json.loads(row["output"]), "rubric": {}}, ensure_ascii=True)
                )
                if row["output"].lstrip().startswith("{")
                else None,
            )
            gates = _failure_gates(fixture_dir)
            self.assertIn("schema_valid_rate", gates)
            self.assertIn("degraded_eval_rate", gates)

    def test_schema_valid_rate_gate_fails_on_a_broken_output(self):
        with self._mutated_fixtures() as fixture_dir:
            self._rewrite(
                fixture_dir / "quiz.jsonl",
                lambda row: row.update(
                    output=json.dumps([{"question": "too short", "type": "nonsense"}])
                ),
            )
            self.assertIn("schema_valid_rate", _failure_gates(fixture_dir))

    def test_prompt_digest_gate_fails_on_an_unbaselined_prompt_edit(self):
        with self._mutated_fixtures() as fixture_dir:
            path = fixture_dir / "prompt_snapshots.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["cases"][sorted(data["cases"])[0]] = "sha256:0000"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertIn("prompt_digest_match_rate", _failure_gates(fixture_dir))

    def test_untrusted_containment_gate_fails_when_a_fence_is_broken(self):
        with self._mutated_fixtures() as fixture_dir:
            self._rewrite(
                fixture_dir / "chat.jsonl",
                lambda row: row["input"].update(
                    recorded_prompt=row["input"]["recorded_prompt"].replace(
                        "</untrusted_input>", "", 1
                    )
                ),
            )
            self.assertIn("untrusted_containment_rate", _failure_gates(fixture_dir))

    def test_prompt_contract_gate_fails_when_a_prompt_drops_a_required_field(self):
        prompt = rebuild_prompt("quiz", {"topics": [{"id": "t", "title": "T", "content": "c"}], "count": 1})
        self.assertEqual(prompt_contract_violations("quiz", prompt), [])
        self.assertEqual(
            prompt_contract_violations("quiz", prompt.replace("correct_answer", "answer_key")),
            ["prompt_omits_required_field:correct_answer"],
        )

    def test_expected_degraded_fixture_that_starts_validating_fails(self):
        with self._mutated_fixtures() as fixture_dir:
            self._rewrite(
                fixture_dir / "interview.jsonl",
                lambda row: row.update(
                    expected={"valid": True}
                )
                if row["case_id"] == "interview.coding.senior.degraded.prose"
                else None,
            )
            self.assertIn("schema_valid_rate", _failure_gates(fixture_dir))

    class _MutatedFixtures:
        def __init__(self, test: unittest.TestCase) -> None:
            self._test = test
            self._tmp = tempfile.TemporaryDirectory()
            self.path = Path(self._tmp.name) / "evals"
            shutil.copytree(EVALS, self.path)

        def __enter__(self) -> Path:
            return self.path

        def __exit__(self, *exc: Any) -> None:  # noqa: ANN401
            self._tmp.cleanup()

    def _mutated_fixtures(self) -> "HarnessMetricsCanFailTests._MutatedFixtures":
        return HarnessMetricsCanFailTests._MutatedFixtures(self)

    @staticmethod
    def _rewrite(path: Path, mutate) -> None:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            mutate(row)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
        )


def _failure_gates(fixture_dir: Path) -> set[str]:
    return {failure["gate"] for failure in run_eval(fixture_dir)["gate_failures"]}


class RetryMetricTests(unittest.TestCase):
    def test_no_run_store_reports_unavailable_not_passing(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = retry_distribution(Path(tmp))
            self.assertFalse(report.available)
            self.assertFalse(report.verified)
            self.assertEqual(report.calls, 0)

    def test_checked_in_retry_metric_is_unverified_until_a_real_run_exists(self):
        distribution = retry_distribution(EVALS)
        self.assertTrue(distribution.available)
        self.assertFalse(
            distribution.verified,
            "only synthetic records exist; a passing retry gate must not be claimed",
        )
        report = run_eval(EVALS)
        self.assertIn("retry_rate", report["unverified_gates"])
        self.assertIn("max_attempts", report["unverified_gates"])

    def test_attempts_are_grouped_from_call_records_not_read_from_a_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp)
            store = fixture_dir / "runs"
            store.mkdir(parents=True)
            rows = [
                {"group_id": "a", "sequence": n, "finish_reason": "stop", "error_code": "",
                 "usage": {"total_tokens": 10}}
                for n in (1, 2, 3)
            ] + [
                {"group_id": "b", "sequence": 1, "finish_reason": "length", "error_code": "llm_truncated",
                 "usage": {"total_tokens": 5}}
            ]
            (store / "llm_calls.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            distribution = retry_distribution(fixture_dir)
            self.assertTrue(distribution.verified)
            self.assertEqual(distribution.calls, 4)
            self.assertEqual(distribution.groups, 2)
            self.assertEqual(distribution.retried_groups, 1)
            self.assertEqual(distribution.retry_rate, 50.0)
            self.assertEqual(distribution.max_attempts, 3)
            self.assertEqual(distribution.attempts_histogram, {"1": 1, "3": 1})
            self.assertEqual(distribution.truncated_calls, 1)
            self.assertEqual(distribution.total_tokens, 35)

    def test_verified_run_failing_the_budget_fails_the_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp)
            store = fixture_dir / "runs"
            store.mkdir(parents=True)
            rows = [
                {"group_id": "a", "sequence": n, "error_code": "llm_call_failed",
                 "finish_reason": "", "usage": {}}
                for n in range(1, 9)
            ]
            (store / "llm_calls.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = run_eval(fixture_dir)
            gates = {failure["gate"] for failure in report["gate_failures"]}
            self.assertIn("max_attempts", gates)
            self.assertIn("retry_rate", gates)
            self.assertEqual(report["unverified_gates"], [])


class FixtureFormatTests(unittest.TestCase):
    def test_avg_retry_count_and_hand_typed_retries_are_rejected(self):
        self.assertEqual(FORBIDDEN_FIXTURE_KEYS, {"retries_used", "avg_retry_count"})
        self.assertNotIn("avg_retry_count", run_eval(EVALS)["summary"])
        self.assertNotIn("avg_retry_count", run_eval(EVALS)["flows"]["questions"])
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "evals"
            fixture_dir.mkdir()
            (fixture_dir / "chat.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "c",
                        "flow": "chat",
                        "input": {"recorded_prompt": "p"},
                        "output": "a reply long enough to pass the length check",
                        "retries_used": 3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            for name in ("questions.jsonl", "quiz.jsonl", "interview_question.jsonl",
                         "interview.jsonl", "custom_topics.jsonl"):
                (fixture_dir / name).write_text("", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_cases(fixture_dir)
            self.assertIn("retries_used", str(ctx.exception))

    def test_every_fixture_carries_an_input_prompt(self):
        for name in harness.FLOW_FILES.values():
            for line in (EVALS / name).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.assertTrue(row.get("input"), f"{name}:{row.get('case_id')} has no input")
                self.assertTrue(
                    rebuild_prompt(row["flow"], row["input"]).strip(),
                    f"{name}:{row.get('case_id')} rebuilds an empty prompt",
                )

    def test_no_fixture_or_run_record_declares_a_metric_by_hand(self):
        for name in harness.FLOW_FILES.values():
            for line in (EVALS / name).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.assertEqual(FORBIDDEN_FIXTURE_KEYS.intersection(json.loads(line)), set())
        for path in (EVALS / "runs").glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.assertEqual(FORBIDDEN_FIXTURE_KEYS.intersection(json.loads(line)), set())

    def test_golden_store_covers_every_fixture_case(self):
        for name in harness.FLOW_FILES.values():
            fixture_ids = {
                json.loads(line)["case_id"]
                for line in (EVALS / name).read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            golden_ids = {
                json.loads(line)["case_id"]
                for line in (EVALS / "golden" / name).read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            self.assertEqual(fixture_ids, golden_ids, f"golden store drift for {name}")

    def test_golden_diff_reports_no_drift_against_the_checked_in_fixtures(self):
        diff = harness.diff_golden(EVALS, EVALS / "golden")
        for flow, entry in diff.items():
            self.assertEqual(entry["status"], "ok", flow)
            self.assertFalse(entry["drift"], f"{flow}: {entry}")


class DeterminismTests(unittest.TestCase):
    def test_harness_runs_without_network_or_provider_credentials(self):
        """No module in the import chain may open a socket or read an API key."""
        import socket

        original = socket.socket

        def forbidden(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            raise AssertionError("the deterministic eval tier must not touch the network")

        socket.socket = forbidden  # type: ignore[assignment]
        try:
            report = run_eval(EVALS)
        finally:
            socket.socket = original  # type: ignore[assignment]
        self.assertEqual(report["tier"], "deterministic")
        self.assertEqual(report["gate_failures"], [])

    def test_report_is_json_serialisable_and_stable(self):
        first = json.dumps(run_eval(EVALS), sort_keys=True)
        second = json.dumps(run_eval(EVALS), sort_keys=True)
        self.assertEqual(first, second)

    def test_cli_json_output_includes_the_report(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exit_code = harness.main(["--fixture-dir", str(EVALS), "--json"])
        self.assertEqual(exit_code, 0)
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["summary"]["total_cases"], len(load_cases(EVALS)))
        self.assertIn("retry_rate", report["summary"]["retry_distribution"])


class InterviewCalibrationTests(unittest.TestCase):
    """Human-graded calibration answers: live tier only, offline never.

    The calibration file is deliberately outside ``FLOW_FILES``, so the
    deterministic tier cannot load it, move a gate, or be blocked by it.
    """

    def test_the_calibration_file_is_not_a_deterministic_flow(self):
        self.assertNotIn(CALIBRATION_FILE, FLOW_CONTRACTS)
        self.assertNotIn(
            CALIBRATION_FILE, [filename for filename in FLOW_CONTRACTS.values()]
        )

    def test_the_checked_in_fixtures_are_well_formed(self):
        rows = load_calibration_fixtures(EVALS)
        self.assertGreaterEqual(len(rows), 8)
        self.assertLessEqual(len(rows), 12)
        seen: set[str] = set()
        for row in rows:
            self.assertTrue(row["case_id"])
            self.assertNotIn(row["case_id"], seen)
            seen.add(row["case_id"])
            self.assertIn(
                row["input"]["session"]["interview_type"],
                {"coding", "behavioral", "mixed", "system_design"},
            )
            self.assertTrue(row["input"]["question"].strip())
            self.assertTrue(row["input"]["answer"].strip())
            self.assertEqual(
                sorted(row["human_scores"]), sorted(CALIBRATION_DIMENSIONS)
            )
            for score in row["human_scores"].values():
                self.assertIsInstance(score, int)
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 5)
            self.assertTrue(row["rationale"].strip())

    def test_the_fixtures_cover_the_system_design_and_coding_dimensions(self):
        rows = load_calibration_fixtures(EVALS)
        types = {row["input"]["session"]["interview_type"] for row in rows}
        self.assertIn("system_design", types)
        self.assertIn("coding", types)
        # At least one answer carries a code contract, so the deterministic
        # grader has something to grade.
        contracts = [
            row["input"].get("question_contract", {}).get("code_contract")
            for row in rows
        ]
        self.assertTrue(any(contracts))
        # At least one answer draws a diagram and one does not.
        diagrams = ["```mermaid" in row["input"]["answer"] for row in rows]
        self.assertTrue(any(diagrams))
        self.assertFalse(all(diagrams))

    def test_the_offline_report_carries_no_calibration_section(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exit_code = harness.main(["--fixture-dir", str(EVALS), "--json"])
        self.assertEqual(exit_code, 0)
        report = json.loads(buffer.getvalue())
        self.assertNotIn("calibration", report)

    def test_a_live_run_reports_per_dimension_deltas_without_touching_a_provider(self):
        """The live replay reports human-vs-model deltas, or nothing at all."""

        class _StubEvaluator:
            def __init__(self):
                self.calls: list[dict] = []

            async def evaluate_answer(self, **kwargs):
                self.calls.append(dict(kwargs))
                return {"rubric": {d: 3 for d in CALIBRATION_DIMENSIONS}}

        evaluator = _StubEvaluator()
        rows = load_calibration_fixtures(EVALS)
        calibration = harness._live_calibration(
            rows, evaluator, provider_configured=True
        )
        self.assertTrue(calibration["available"])
        self.assertEqual(calibration["answers"], len(rows))
        self.assertEqual(len(evaluator.calls), len(rows))
        # Every dimension is graded one point away from the human's score, so
        # the row records the delta rather than only the absolute numbers.
        first = calibration["rows"][0]
        # The stub scores a flat 3 everywhere, so the delta is the distance
        # between that and the human grade, not an absolute score.
        self.assertEqual(
            first["deltas"]["technical_accuracy"],
            3 - first["human_scores"]["technical_accuracy"],
        )
        self.assertIn(first["deltas"]["technical_accuracy"], range(-5, 6))
        self.assertTrue(
            all(abs(d) <= 3 for d in first["deltas"].values())
        )
        self.assertIsInstance(first["within_one_point"], bool)
        self.assertGreaterEqual(calibration["agreement_within_one_point"], 0.0)
        # The answer is replayed with the contract that produced it.
        self.assertIn("question_contract", evaluator.calls[0])

    def test_a_live_per_answer_failure_does_not_cost_the_report(self):
        class _BrokenEvaluator:
            async def evaluate_answer(self, **kwargs):
                raise RuntimeError("provider 500")

        calibration = harness._live_calibration(
            load_calibration_fixtures(EVALS),
            _BrokenEvaluator(),
            provider_configured=True,
        )
        self.assertTrue(calibration["available"])
        self.assertEqual(calibration["answers"], 0)
        self.assertEqual(len(calibration["errors"]), len(load_calibration_fixtures(EVALS)))

    def test_a_malformed_calibration_row_fails_loudly(self):
        row = {
            "case_id": "bad",
            "input": {"session": {}, "question": "q", "answer": "a"},
            "human_scores": {"technical_accuracy": 7},
            "rationale": "r",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / CALIBRATION_FILE
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibration_fixtures(Path(tmp))

    def test_missing_dimension_fails_loudly(self):
        row = {
            "case_id": "bad",
            "input": {"session": {}, "question": "q", "answer": "a"},
            "human_scores": {"technical_accuracy": 3},
            "rationale": "r",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / CALIBRATION_FILE
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibration_fixtures(Path(tmp))

    def test_comparison_reports_deltas_and_agreement(self):
        human = {d: 3 for d in CALIBRATION_DIMENSIONS}
        model = {d: 3 for d in CALIBRATION_DIMENSIONS}
        result = compare_calibration(human, model)
        self.assertEqual(result["mean_absolute_error"], 0.0)
        self.assertTrue(result["within_one_point"])

    def test_comparison_flags_a_disagreement_outside_one_point(self):
        human = {d: 2 for d in CALIBRATION_DIMENSIONS}
        model = {
            "technical_accuracy": 5,
            "reasoning_depth": 2,
            "communication_clarity": 2,
            "completeness": 2,
            "confidence_signal": 2,
        }
        result = compare_calibration(human, model)
        self.assertEqual(result["deltas"]["technical_accuracy"], 3)
        self.assertFalse(result["within_one_point"])
        self.assertAlmostEqual(result["mean_absolute_error"], 0.6)


if __name__ == "__main__":
    unittest.main()
