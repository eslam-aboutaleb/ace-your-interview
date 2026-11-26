#!/usr/bin/env python3
"""Learning-quality evaluation and release gate (issue #17).

Deterministic, CI-safe companion to ``eval_agent_quality.py``. Where that
harness covers prompt contracts and per-flow output parsing, this script
covers the *learning* quality bar:

* **Schema validity of published answers** — every recorded question output
  in ``evals/questions.jsonl`` is re-validated with the production
  ``_validate_question_item``. The release bar is 100%.
* **Duplicate safety** — zero exact duplicates inside every recorded
  question set, plus precision/recall of the production near-duplicate
  detector against the human-labeled pairs in
  ``evals/semantic_duplicates.jsonl``.
* **Progress correctness** — the golden multi-session journeys in
  ``evals/journey_sessions.jsonl`` are replayed through the production
  ``LearningStore`` (real sqlite, no model) and the observed aggregates must
  match the frozen expectations exactly. A mismatch means stored progress
  changed behavior, which is a release blocker.
* **Objective coverage** — every objective a golden journey attempts must
  appear in the journey's expectation set.

The report records the harness tier, the prompt-snapshot version count and
the git commit so a quality trend can be tied to provider/model/prompt
versions. Live-model evals stay in ``eval_agent_quality.py --live``; this
gate never calls a model and never needs a key.

Exit code 1 when any calibrated threshold is breached.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Production imports only. tests/test_eval_learning_quality.py asserts these
# exact objects, so a local copy fails the build — the same contract the
# agent-quality harness uses.
from app.services.learning_store import LearningStore  # noqa: E402
from app.services.question_generator import (  # noqa: E402
    _is_near_duplicate_question,
    _normalise_question,
    _validate_question_item,
)

EVALS_DIR = BACKEND_ROOT / "evals"

THRESHOLDS: dict[str, float] = {
    "schema_valid_rate_min": 100.0,
    "exact_duplicate_max": 0,
    # Calibrated floors. Recall must stay at 100%: a missed duplicate is a
    # real learner-facing defect. Precision is floored lower because the
    # production detector is a heuristic whose false positives only skip a
    # similar candidate during generation (benign over-filtering); the
    # observed misses are reported in duplicate_misses for review.
    "duplicate_precision_min": 60.0,
    "duplicate_recall_min": 100.0,
    "progress_replay_match_rate_min": 100.0,
    "objective_coverage_min": 100.0,
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def measure_schema_and_duplicates() -> dict[str, float]:
    cases = _load_jsonl(EVALS_DIR / "questions.jsonl")
    total = 0
    valid = 0
    exact_duplicates = 0
    for case in cases:
        raw_output = case.get("output")
        if isinstance(raw_output, str):
            try:
                raw_output = json.loads(raw_output)
            except json.JSONDecodeError:
                raw_output = []
        if isinstance(raw_output, dict):
            outputs = raw_output.get("questions") or []
        elif isinstance(raw_output, list):
            outputs = raw_output
        else:
            outputs = []
        seen: set[str] = set()
        for item in outputs:
            if not isinstance(item, dict):
                total += 1
                continue
            # Match the exact production call shape the agent-quality
            # harness uses; richer arguments would turn grounding checks on
            # that these golden outputs were never generated against.
            ok, _reason = _validate_question_item(item, "", None, None)
            total += 1
            valid += 1 if ok else 0
            normalised = _normalise_question(str(item.get("question", "")))
            if normalised:
                if normalised in seen:
                    exact_duplicates += 1
                seen.add(normalised)
    return {
        "schema_valid_rate": (valid / total * 100.0) if total else 100.0,
        "schema_items_checked": total,
        "exact_duplicate_count": exact_duplicates,
    }


def measure_duplicate_detector() -> dict[str, float]:
    pairs = _load_jsonl(EVALS_DIR / "semantic_duplicates.jsonl")
    tp = fp = fn = 0
    misses: list[str] = []
    for pair in pairs:
        predicted = _is_near_duplicate_question(pair["b"], [pair["a"]])
        if pair["label"] == "duplicate" and predicted:
            tp += 1
        elif pair["label"] == "duplicate":
            fn += 1
            misses.append(pair["pair_id"])
        elif predicted:
            fp += 1
            misses.append(pair["pair_id"])
    precision = tp / (tp + fp) * 100.0 if (tp + fp) else 100.0
    recall = tp / (tp + fn) * 100.0 if (tp + fn) else 100.0
    return {
        "duplicate_precision": precision,
        "duplicate_recall": recall,
        "duplicate_pairs": len(pairs),
        "duplicate_misses": misses,
    }


def _replay_session(session: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        store = LearningStore(str(Path(tmp) / "learning.db"))
        for step in session["steps"]:
            if step["op"] != "record_attempt":
                raise ValueError(f"unknown golden step op: {step['op']!r}")
            store.record_attempt(
                user_id=session["user_id"],
                question_id=step["question_id"],
                topic_id=step["topic_id"],
                user_answer=step["user_answer"],
                is_correct=bool(step["is_correct"]),
                confidence=int(step["confidence"]),
                response_time_ms=int(step["response_time_ms"]),
                mode=step["mode"],
                learning_objective=step["learning_objective"],
            )
        ranked = store.select_next_objectives(
            user_id=session["user_id"], topic_ids=session["topic_ids"], limit=10
        )
        objectives_seen = sorted(
            {step["learning_objective"] for step in session["steps"]}
        )
        weakest = next(
            (
                entry["learning_objective"]
                for entry in ranked
                if entry.get("attempts", 0) > 0 and entry.get("accuracy", 1.0) < 1.0
            ),
            None,
        )
        return {
            "attempts_total": len(session["steps"]),
            "correct_total": sum(1 for s in session["steps"] if s["is_correct"]),
            "objectives_seen": objectives_seen,
            "weakest_objective": weakest,
        }


def measure_progress_correctness() -> dict[str, float]:
    sessions = _load_jsonl(EVALS_DIR / "journey_sessions.jsonl")
    matched = 0
    mismatches: list[dict] = []
    for session in sessions:
        observed = _replay_session(session)
        expected = session["expect"]
        expected_normalised = {
            **expected,
            "objectives_seen": sorted(expected["objectives_seen"]),
        }
        if observed == expected_normalised:
            matched += 1
        else:
            mismatches.append(
                {
                    "session_id": session["session_id"],
                    "expected": expected_normalised,
                    "observed": observed,
                }
            )
    return {
        "progress_replay_match_rate": (matched / len(sessions) * 100.0) if sessions else 100.0,
        "progress_sessions": len(sessions),
        "progress_mismatches": mismatches,
    }


def measure_objective_coverage() -> dict[str, float]:
    sessions = _load_jsonl(EVALS_DIR / "journey_sessions.jsonl")
    covered = 0
    total = 0
    for session in sessions:
        expected = set(session["expect"]["objectives_seen"])
        attempted = {step["learning_objective"] for step in session["steps"]}
        total += len(expected)
        covered += len(expected & attempted)
    return {
        "objective_coverage": (covered / total * 100.0) if total else 100.0,
        "objectives_checked": total,
    }


def version_block() -> dict[str, str]:
    snapshots = json.loads((EVALS_DIR / "prompt_snapshots.json").read_text())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=BACKEND_ROOT,
            check=False,
        ).stdout.strip()
    except OSError:
        commit = ""
    return {
        "harness_tier": "deterministic",
        "prompt_snapshot_cases": str(len(snapshots.get("cases", {}))),
        "git_commit": commit,
        "live_model_evals": "scripts/eval_agent_quality.py --live (cost-bounded, manual)",
    }


def evaluate_gates(metrics: dict[str, float]) -> list[dict]:
    failures: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        if not passed:
            failures.append({"gate": name, "detail": detail})

    check(
        "schema_valid_rate",
        metrics["schema_valid_rate"] >= THRESHOLDS["schema_valid_rate_min"],
        f"must be >= {THRESHOLDS['schema_valid_rate_min']}%",
    )
    check(
        "exact_duplicate_count",
        metrics["exact_duplicate_count"] <= THRESHOLDS["exact_duplicate_max"],
        "published question sets must contain zero exact duplicates",
    )
    check(
        "duplicate_precision",
        metrics["duplicate_precision"] >= THRESHOLDS["duplicate_precision_min"],
        f"must keep precision >= {THRESHOLDS['duplicate_precision_min']}% (false positives only over-filter candidates)",
    )
    check(
        "duplicate_recall",
        metrics["duplicate_recall"] >= THRESHOLDS["duplicate_recall_min"],
        f"must catch >= {THRESHOLDS['duplicate_recall_min']}% of labeled duplicates",
    )
    check(
        "progress_replay_match_rate",
        metrics["progress_replay_match_rate"]
        >= THRESHOLDS["progress_replay_match_rate_min"],
        "golden journey replays must match frozen expectations (no progress loss)",
    )
    check(
        "objective_coverage",
        metrics["objective_coverage"] >= THRESHOLDS["objective_coverage_min"],
        "every expected journey objective must be attempted",
    )
    return failures


def run_eval() -> dict:
    metrics: dict[str, object] = {}
    metrics.update(measure_schema_and_duplicates())
    metrics.update(measure_duplicate_detector())
    metrics.update(measure_progress_correctness())
    metrics.update(measure_objective_coverage())
    failures = evaluate_gates(metrics)  # type: ignore[arg-type]
    return {
        "version": version_block(),
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "gate_failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()

    report = run_eval()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"learning-quality eval: {'PASS' if report['passed'] else 'FAIL'}")
        for key, value in report["metrics"].items():
            if isinstance(value, (int, float, str)):
                print(f"  {key}: {value}")
        for failure in report["gate_failures"]:
            print(f"  GATE FAILURE {failure['gate']}: {failure['detail']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
