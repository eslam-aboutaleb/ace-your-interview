#!/usr/bin/env python3
"""Offline evaluation harness for prompt/output quality checks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

FLOW_FILES = {
    "questions": "questions.jsonl",
    "quiz": "quiz.jsonl",
    "interview": "interview.jsonl",
    "chat": "chat.jsonl",
    "custom_topics": "custom_topics.jsonl",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", _norm(text))
        if len(t) >= 4
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _validate_questions(row: dict[str, Any]) -> tuple[bool, int, int, int]:
    items = row.get("items")
    if not isinstance(items, list) or not items:
        return False, 0, 0, 0
    seen: set[str] = set()
    duplicates = 0
    grounded = 0
    source_tokens = _tokens(str(row.get("source_text", "")))
    for item in items:
        if not isinstance(item, dict):
            return False, duplicates, grounded, len(items)
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        difficulty = str(item.get("difficulty", "")).strip().lower()
        if not question or not answer or difficulty not in {"easy", "medium", "hard"}:
            return False, duplicates, grounded, len(items)
        q_norm = _norm(question)
        if q_norm in seen:
            duplicates += 1
        seen.add(q_norm)
        quote_tokens = _tokens(str(item.get("source_quote", "")))
        if quote_tokens and source_tokens and (quote_tokens & source_tokens):
            grounded += 1
    return True, duplicates, grounded, len(items)


def _validate_quiz(row: dict[str, Any]) -> tuple[bool, int, int, int]:
    items = row.get("items")
    if not isinstance(items, list) or not items:
        return False, 0, 0, 0
    seen: set[str] = set()
    duplicates = 0
    grounded = 0
    source_tokens = _tokens(str(row.get("source_text", "")))
    for item in items:
        if not isinstance(item, dict):
            return False, duplicates, grounded, len(items)
        q = str(item.get("question", "")).strip()
        qtype = str(item.get("type", "")).strip().lower()
        choices = item.get("choices")
        correct = str(item.get("correct_answer", "")).strip()
        if not q or qtype not in {"mcq", "true_false"} or not isinstance(choices, list) or not correct:
            return False, duplicates, grounded, len(items)
        q_norm = _norm(q)
        if q_norm in seen:
            duplicates += 1
        seen.add(q_norm)
        quote_tokens = _tokens(str(item.get("source_quote", "")))
        if quote_tokens and source_tokens and (quote_tokens & source_tokens):
            grounded += 1
    return True, duplicates, grounded, len(items)


def _validate_interview(row: dict[str, Any]) -> tuple[bool, int]:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return False, 0
    rubric = payload.get("rubric")
    if not isinstance(rubric, dict):
        return False, 0
    required = {
        "technical_accuracy",
        "reasoning_depth",
        "communication_clarity",
        "completeness",
        "confidence_signal",
        "overall",
    }
    if not required.issubset(set(rubric.keys())):
        return False, 0
    return True, 1


def _validate_chat(row: dict[str, Any]) -> tuple[bool, int]:
    reply = str(row.get("reply", "")).strip()
    return bool(reply and len(reply) >= 20), 1


def _validate_custom_topic(row: dict[str, Any]) -> tuple[bool, int]:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return False, 0
    sections = payload.get("sections")
    if not isinstance(sections, list) or len(sections) < 100:
        return False, 0
    return True, 1


def run_eval(fixture_dir: Path) -> dict[str, Any]:
    total_cases = 0
    valid_cases = 0
    total_retries = 0
    total_items = 0
    total_duplicates = 0
    grounded_items = 0
    groundable_items = 0
    per_flow: dict[str, dict[str, Any]] = {}

    for flow, filename in FLOW_FILES.items():
        rows = _read_jsonl(fixture_dir / filename)
        flow_total = 0
        flow_valid = 0
        for row in rows:
            retries = int(row.get("retries_used", 0) or 0)
            total_retries += max(0, retries)
            flow_total += 1
            total_cases += 1
            if flow == "questions":
                ok, dup, grounded, item_count = _validate_questions(row)
                total_duplicates += dup
                total_items += item_count
                grounded_items += grounded
                groundable_items += item_count
            elif flow == "quiz":
                ok, dup, grounded, item_count = _validate_quiz(row)
                total_duplicates += dup
                total_items += item_count
                grounded_items += grounded
                groundable_items += item_count
            elif flow == "interview":
                ok, count = _validate_interview(row)
                total_items += count
            elif flow == "chat":
                ok, count = _validate_chat(row)
                total_items += count
            else:
                ok, count = _validate_custom_topic(row)
                total_items += count

            if ok:
                flow_valid += 1
                valid_cases += 1

        per_flow[flow] = {
            "total": flow_total,
            "valid": flow_valid,
            "schema_valid_rate": round((flow_valid / flow_total) * 100, 2) if flow_total else 0.0,
        }

    schema_valid_rate = round((valid_cases / total_cases) * 100, 2) if total_cases else 0.0
    duplicate_rate = round((total_duplicates / total_items) * 100, 2) if total_items else 0.0
    grounding_fidelity = round((grounded_items / groundable_items) * 100, 2) if groundable_items else 0.0
    avg_retries = round(total_retries / total_cases, 2) if total_cases else 0.0
    return {
        "summary": {
            "total_cases": total_cases,
            "schema_valid_rate": schema_valid_rate,
            "avg_retry_count": avg_retries,
            "duplicate_rate": duplicate_rate,
            "grounding_fidelity": grounding_fidelity,
        },
        "flows": per_flow,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline eval for agent quality metrics.")
    parser.add_argument(
        "--fixture-dir",
        default=str(Path(__file__).resolve().parents[1] / "evals"),
        help="Directory with JSONL eval fixtures.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    args = parser.parse_args()

    report = run_eval(Path(args.fixture_dir))
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0

    summary = report["summary"]
    print("Agent Quality Eval")
    print(f"- total_cases: {summary['total_cases']}")
    print(f"- schema_valid_rate: {summary['schema_valid_rate']}%")
    print(f"- avg_retry_count: {summary['avg_retry_count']}")
    print(f"- duplicate_rate: {summary['duplicate_rate']}%")
    print(f"- grounding_fidelity: {summary['grounding_fidelity']}%")
    for flow, data in report["flows"].items():
        print(f"- {flow}: {data['valid']}/{data['total']} ({data['schema_valid_rate']}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
