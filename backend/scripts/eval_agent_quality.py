#!/usr/bin/env python3
"""Evaluation harness for prompt-contract and output quality (Stage 4.1).

Two tiers, one report shape:

* **Deterministic (CI).** ``run_eval`` reads the fixtures in ``evals/`` and
  re-runs the *production* parsers and validators over each recorded output. It
  also rebuilds each flow's *input prompt* from the fixture's input using the
  production prompt builders, so a prompt edit that drops a required output
  field is detected here. No network, no LLM calls, no API keys.
* **Live (manual / scheduled).** ``--live`` replays the same fixture inputs
  against the real generators through ``LLMClient``, records per-call telemetry
  (``usage`` / ``finish_reason`` / ``error_code``) into ``evals/runs/`` and
  diffs the fresh output against the checked-in golden store in ``evals/golden/``.

Deliberate non-goals:

* The harness owns **no** validators of its own. Anything that decides whether
  an output is acceptable is imported from ``app.services``; a local copy is the
  bug this file exists to remove (it used to report ``schema_valid_rate: 100.0``
  while production disagreed).
* Retry counts are **never** read from a fixture. They are derived by grouping
  the recorded ``LLMClient`` call records of an instrumented run, so a hand-typed
  ``retries_used`` in a JSONL file cannot inflate or deflate the number.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# ── Production imports (the point of Stage 4.1) ────────────────────────────
# Every validator, parser, prompt builder and normaliser below is the one that
# runs in production. ``tests/test_eval_agent_quality.py`` asserts these exact
# objects are the harness's, so replacing one with a local copy fails the build.
#
# They come after the sys.path bootstrap so ``python scripts/eval_agent_quality.py``
# works from a bare checkout with nothing installed.
from app.services.interview_generator import InterviewGenerator  # noqa: E402
from app.services.llm_client import parse_json_object  # noqa: E402
from app.services.prompt_blocks import (  # noqa: E402
    CLOSE_TAG,
    OPEN_TAG,
    _FENCE_TOKENS,
    neutralise_fence,
    render_contract,
    untrusted_block,
)
from app.services.question_generator import (  # noqa: E402
    _attempt_budget,
    _build_prompt,
    _build_quiz_prompt,
    _grounding_anchor_phrases,
    _grounding_tokens,
    _parse_questions_json,
    _validate_question_item,
    _validate_quiz_item,
)
from app.services.question_text import normalise_question  # noqa: E402
from app.services.topic_catalog import PROBLEM_SOLVING_LANGUAGE_OPTIONS  # noqa: E402

#: Import-time aliases the tests bind to. A local re-implementation cannot be
#: substituted for any of these without breaking ``test_harness_binds_production_*``.
PRODUCTION_VALIDATORS: dict[str, Any] = {
    "parse_json_object": parse_json_object,
    "parse_questions_json": _parse_questions_json,
    "validate_question_item": _validate_question_item,
    "validate_quiz_item": _validate_quiz_item,
    "validate_interview_question_payload": InterviewGenerator._validate_question_payload,
    "validate_interview_eval_payload": InterviewGenerator._validate_eval_payload,
    "normalise_eval_payload": InterviewGenerator._normalise_eval_payload,
    "normalise_question": normalise_question,
    "render_contract": render_contract,
    "untrusted_block": untrusted_block,
    "neutralise_fence": neutralise_fence,
    "attempt_budget": _attempt_budget,
}

#: Fields that must never appear in a fixture or run record. ``retries_used``
#: was read as if it were a measurement; keeping it out of the format makes a
#: hand-typed retry count impossible rather than merely discouraged.
FORBIDDEN_FIXTURE_KEYS = {"retries_used", "avg_retry_count"}

#: ``_normalise_eval_payload`` and the repair helpers behind it are instance
#: methods that touch no instance state (every collaborator is a ``staticmethod``),
#: so an un-initialised instance calls the real production code without needing a
#: live ``LLMClient``/``DocParser``. Running ``__init__`` would make the
#: deterministic tier require a provider client it must never touch.
_INTERVIEW_EVAL_NORMALISER = InterviewGenerator.__new__(InterviewGenerator)

FLOW_FILES: dict[str, str] = {
    "questions": "questions.jsonl",
    "quiz": "quiz.jsonl",
    "interview_question": "interview_question.jsonl",
    "interview": "interview.jsonl",
    "chat": "chat.jsonl",
    "custom_topics": "custom_topics.jsonl",
}

#: Reviewable policy, not measurement. Every one of these is reachable by an
#: ordinary prompt or model regression; none of them is a tautology.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "schema_valid_rate_min": 100.0,
    "prompt_contract_rate_min": 100.0,
    "prompt_output_field_rate_min": 100.0,
    "untrusted_containment_rate_min": 100.0,
    "prompt_digest_match_rate_min": 100.0,
    "duplicate_rate_max": 0.0,
    "grounding_fidelity_min": 50.0,
    "repair_rate_max": 50.0,
    "degraded_eval_rate_max": 25.0,
    "retry_rate_max": 25.0,
    "max_attempts_max": 3.0,
}

RUN_STORE_NAME = "llm_calls.jsonl"
SYNTHETIC_RUN_STORE_NAME = "llm_calls.synthetic.jsonl"


# ── small local helpers (measurement only, never acceptance) ────────────────


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{lineno} is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{path.name}:{lineno} is not a JSON object")
        rows.append(parsed)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


# ── required-output-field extraction, derived from the validators ──────────

_MISSING_ISSUE = re.compile(r"^(?:missing_or_empty|missing)_([a-z][a-z0-9_]*)$")

#: Minimal filler that satisfies a validator's *next* complaint without
#: inventing a requirement: each entry is the smallest value that advances the
#: production validator one step. Nothing here decides acceptance — it only lets
#: the harvest loops below walk the validator's own required-field surface.
_HARVEST_FILLERS: dict[str, Any] = {
    "question": "Why does idempotency matter for retry safety in this service?",
    "answer": "Because repeating the same request must not create duplicate mutations.",
    "rubric": {
        "technical_accuracy": 4,
        "reasoning_depth": 4,
        "communication_clarity": 4,
        "completeness": 4,
        "confidence_signal": 4,
        "overall": 80,
    },
    "strengths": ["States the tradeoff clearly."],
    "improvements": ["Quantify the expected impact."],
    "follow_up_note": "### What strong interviewers wanted to hear\nClear framing.\n",
    "competency_focus": "Retry safety",
    "expected_signals": ["deduplication key", "bounded retry"],
    "type": "mcq",
    "choices": [
        {"label": "A", "text": "first"},
        {"label": "B", "text": "second"},
        {"label": "C", "text": "third"},
        {"label": "D", "text": "fourth"},
    ],
    "correct_answer": "A",
    "explanation": "An explanation long enough to pass the validator.",
    "difficulty": "medium",
    "topic_id": "eval-harvest-topic",
    "source_quote": "A source quote taken from the course content.",
    "reasoning_summary": "A reasoning summary for the harvested item.",
    "target_level": "mid",
}

_MAX_HARVEST_STEPS = 32


def _clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (dict(value) if isinstance(value, dict) else (list(value) if isinstance(value, list) else value))
        for key, value in payload.items()
    }


def _sufficiency_harvest(
    validate: Callable[[dict[str, Any]], tuple[bool, str]], required: set[str]
) -> dict[str, Any]:
    """Fill exactly the fields the validator complains about, one complaint at a time."""
    payload: dict[str, Any] = {}
    for _ in range(_MAX_HARVEST_STEPS):
        _, issue = validate(_clone_payload(payload))
        match = _MISSING_ISSUE.match(issue)
        if not match:
            break
        field_name = match.group(1)
        if field_name in required:
            break
        filler = _HARVEST_FILLERS.get(field_name)
        required.add(field_name)
        if filler is None:
            # An unmapped requirement: record it (so the contract widens and the
            # report shows it) but do not pretend we know how to satisfy it.
            break
        payload[field_name] = _clone_payload(filler) if isinstance(filler, dict) else filler
    return payload


def _necessity_harvest(
    validate: Callable[[dict[str, Any]], tuple[bool, str]],
    payload: dict[str, Any],
    required: set[str],
    path: tuple[str, ...] = (),
) -> None:
    """Drop each field in turn; a field the validator refuses without is required.

    Order-independent, so it also catches requirements hidden behind an earlier
    check that returns a non-``missing_*`` code first (``question_too_short``,
    for instance, is reported before ``competency_focus``). ``path`` walks into a
    nested object — ``rubric`` is validated through its wrapper, so the probe
    removes a subfield in place rather than validating the sub-object alone.
    """
    target: Any = payload
    for step in path:
        target = target[step]
    for key, value in list(target.items()):
        without = _clone_payload(payload)
        cursor: Any = without
        for step in path:
            cursor = cursor[step]
        del cursor[key]
        ok, issue = validate(without)
        if ok:
            continue
        match = _MISSING_ISSUE.match(issue)
        if not match or match.group(1) != key:
            continue
        required.add(key)
        if isinstance(value, dict):
            _necessity_harvest(validate, payload, required, path + (key,))


def _seed_payload(
    validate: Callable[[dict[str, Any]], tuple[bool, str]], required: set[str]
) -> dict[str, Any]:
    """A payload the validator accepts outright, so necessity probing can run.

    ``_sufficiency_harvest`` stops early whenever a validator reports a
    non-``missing_*`` issue before its required-field checks (a too-short
    question, say). When that happens, fall back to the full filler payload; if
    the validator accepts it, the necessity pass can still walk its real
    required surface.
    """
    payload = _sufficiency_harvest(validate, required)
    if validate(_clone_payload(payload))[0]:
        return payload
    full = _clone_payload(_HARVEST_FILLERS)
    return full if validate(_clone_payload(full))[0] else payload


def _required_output_fields(validate: Callable[[dict[str, Any]], tuple[bool, str]]) -> frozenset[str]:
    """Field names a production validator refuses to work without.

    Both a sufficiency pass (fill what the validator asks for) and a necessity
    pass (remove what the validator insists on) run against the *production*
    validator, so whatever that validator requires is the required set. Nothing
    is transcribed by hand: a validator that grows a required field immediately
    widens the prompt contract the harness enforces, and a prompt that stops
    naming that field fails the build.
    """
    required: set[str] = set()
    payload = _seed_payload(validate, required)
    if validate(_clone_payload(payload))[0]:
        _necessity_harvest(validate, payload, required)
    return frozenset(required)


@dataclass(frozen=True)
class FlowContract:
    """The output shape a flow's production validator requires."""

    flow: str
    probe: Callable[[dict[str, Any]], tuple[bool, str]]
    fields: frozenset[str]


FLOW_CONTRACTS: dict[str, FlowContract] = {}


def _question_item_contract(payload: dict[str, Any]) -> tuple[bool, str]:
    # A non-problem-solving topic so the probe measures only the required-field
    # surface, not the problem-solving heading/fence rules.
    return _validate_question_item(payload, "", None, None)


def _quiz_item_contract(payload: dict[str, Any]) -> tuple[bool, str]:
    return _validate_quiz_item(payload, {"eval-harvest-topic"}, {"mcq", "true_false"}, None, None)


def _interview_question_contract(payload: dict[str, Any]) -> tuple[bool, str]:
    return InterviewGenerator._validate_question_payload(payload, [])


def _interview_eval_contract(payload: dict[str, Any]) -> tuple[bool, str]:
    return InterviewGenerator._validate_eval_payload(payload)


def _build_flow_contracts() -> dict[str, FlowContract]:
    probes: dict[str, Callable[[dict[str, Any]], tuple[bool, str]]] = {
        "questions": _question_item_contract,
        "quiz": _quiz_item_contract,
        "interview_question": _interview_question_contract,
        "interview": _interview_eval_contract,
    }
    return {
        flow: FlowContract(flow=flow, probe=probe, fields=_required_output_fields(probe))
        for flow, probe in probes.items()
    }


FLOW_CONTRACTS = _build_flow_contracts()


# ── prompt rebuilding (production builders, no network) ─────────────────────


def _interview_prompt_parts(case_input: dict[str, Any]) -> tuple[str, str]:
    """Return ``(question_prompt, evaluate_prompt)`` for an interview case.

    Both builders are ``@staticmethod`` on ``InterviewGenerator`` and touch no
    instance state, so they can be called without constructing a generator
    (which would need a live ``LLMClient``).
    """
    session = dict(case_input.get("session") or {})
    turns = list(case_input.get("turns") or [])
    question_prompt = InterviewGenerator._question_prompt(session, turns)
    evaluate_prompt = InterviewGenerator._evaluate_prompt(
        session,
        str(case_input.get("question", "")),
        str(case_input.get("answer", "")),
        int(case_input.get("turn_index", 0) or 0),
    )
    return question_prompt, evaluate_prompt


def rebuild_prompt(flow: str, case_input: dict[str, Any]) -> str:
    """Render the production input prompt for a fixture's input.

    ``chat`` has no pure builder (its prompt is assembled inline in the router
    handler), so its fixture carries the recorded prompt instead.
    """
    if flow == "questions":
        return _build_prompt(
            topic_id=str(case_input.get("topic_id", "")),
            topic_title=str(case_input.get("topic_title", "")),
            doc_content=str(case_input.get("doc_content", "")),
            count=int(case_input.get("count", 5) or 5),
            difficulty=case_input.get("difficulty"),
            level=case_input.get("level"),
            section_title=case_input.get("section_title"),
            section_content=case_input.get("section_content"),
            response_detail=str(case_input.get("response_detail", "very_detailed")),
            preferred_language=str(case_input.get("preferred_language", "")),
            requires_programming=bool(case_input.get("requires_programming", False)),
            requested_total_count=case_input.get("requested_total_count"),
            existing_questions=list(case_input.get("existing_questions") or []),
            additional_existing_questions=list(
                case_input.get("additional_existing_questions") or []
            ),
            prior_progress=str(case_input.get("prior_progress", "")),
            mcp_context=str(case_input.get("mcp_context", "")),
        )
    if flow == "quiz":
        return _build_quiz_prompt(
            topics_content=[dict(topic) for topic in case_input.get("topics") or []],
            count=int(case_input.get("count", 10) or 10),
            question_types=list(case_input.get("question_types") or []) or None,
            difficulty=case_input.get("difficulty"),
            level=case_input.get("level"),
            response_detail=str(case_input.get("response_detail", "concise")),
            preferred_language=str(case_input.get("preferred_language", "")),
            existing_questions=list(case_input.get("existing_questions") or []),
            additional_existing_questions=list(
                case_input.get("additional_existing_questions") or []
            ),
            prior_progress=str(case_input.get("prior_progress", "")),
            mcp_context=str(case_input.get("mcp_context", "")),
        )
    if flow == "interview_question":
        question_prompt, _ = _interview_prompt_parts(case_input)
        return question_prompt
    if flow == "interview":
        _, evaluate_prompt = _interview_prompt_parts(case_input)
        return evaluate_prompt
    if flow == "custom_topics":
        from app.services.custom_topic_generator import _build_stream_batch_prompt

        return _build_stream_batch_prompt(
            topic=str(case_input.get("topic", "")),
            target_sections=int(case_input.get("target_sections", 120) or 120),
            start_index=int(case_input.get("start_index", 0) or 0),
            batch_size=int(case_input.get("batch_size", 10) or 10),
            existing_headings=[str(h) for h in case_input.get("existing_headings") or []],
            mcp_context=str(case_input.get("mcp_context", "")),
        )
    if flow == "chat":
        return str(case_input.get("recorded_prompt", ""))
    raise KeyError(f"no prompt builder for flow {flow!r}")


def prompt_contract_violations(flow: str, prompt: str) -> list[str]:
    """Production-derived reasons this prompt cannot yield a valid output.

    Each production validator names the fields it refuses to work without. If a
    prompt edit stops naming one of them, the model cannot be asked for it and
    the flow's parse rate collapses — so the omission is the regression.
    """
    if not prompt.strip():
        return ["empty_prompt"]
    contract = FLOW_CONTRACTS.get(flow)
    if contract is None:
        return []
    lowered = prompt.lower()
    return [f"prompt_omits_required_field:{name}" for name in sorted(contract.fields) if name not in lowered]


def prompt_output_field_violations(
    flow: str, prompt: str, raw_output: str, valid: bool
) -> list[str]:
    """Fields the model emitted that the prompt never asked for.

    The mirror image of :func:`prompt_contract_violations`: instead of asking
    "does the prompt name everything the validator requires", this asks "does
    the prompt name everything the model actually produced". A field that shows
    up in real output without appearing in the contract is a guess the prompt
    failed to specify, which is exactly the class of defect a golden snapshot
    cannot catch on its own.

    Skipped for cases the production validator already rejects: those outputs
    are malformed by construction and their keys mean nothing.
    """
    if not valid or not prompt.strip():
        return []
    keys = _output_field_names(flow, raw_output)
    if not keys:
        return []
    lowered = prompt.lower()
    return [f"prompt_omits_emitted_field:{name}" for name in sorted(keys) if name not in lowered]


def validate(flow: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Run a flow's production validator over one candidate.

    Returns the validator's verdict together with the item as production left it,
    because these validators normalise in place and the normalised form is what
    the rest of the harness measures. Callers pass their own copy.
    """
    ok, _issue = FLOW_CONTRACTS[flow].probe(payload)
    return ok, payload


@lru_cache(maxsize=None)
def server_assigned_fields(flow: str) -> frozenset[str]:
    """Fields a flow's production validator writes itself rather than requiring.

    ``_validate_question_item`` stamps ``topic_id``, ``source_section`` and the
    other auxiliary fields onto every item whatever the model returned, so those
    keys are server-assigned: blaming the prompt for not naming them would flag a
    contract that never existed. Derived by running the production validator on
    two differently-filled payloads and keeping the keys whose output value is
    identical both times.
    """
    probe = _seed_payload(FLOW_CONTRACTS[flow].probe, set())
    alternate = {
        key: (
            {sub: f"alternate {sub} value" for sub in value}
            if isinstance(value, dict)
            else (["alternate one", "alternate two"] if isinstance(value, list) else f"alternate {key} value")
        )
        for key, value in probe.items()
    }
    if not probe or not validate(flow, _clone_payload(probe))[0]:
        return frozenset()
    if not validate(flow, _clone_payload(alternate))[0]:
        return frozenset()
    first = validate(flow, _clone_payload(probe))[1]
    second = validate(flow, _clone_payload(alternate))[1]
    return frozenset(
        key
        for key in first
        if key in second
        and first[key] == second[key]
        and probe.get(key) != first[key]
    )


def _output_field_names(flow: str, raw_output: str) -> set[str]:
    names: set[str] = set()
    if flow in {"questions", "quiz"}:
        for item in _parse_questions_json(raw_output):
            if isinstance(item, dict):
                names.update(str(key) for key in item)
    elif flow == "interview_question":
        payload = parse_json_object(raw_output)
        names.update(str(key) for key in payload)
    elif flow == "interview":
        payload = parse_json_object(raw_output)
        rubric = payload.get("rubric")
        if isinstance(rubric, dict):
            names.update(str(key) for key in rubric)
        names.update(str(key) for key in payload if key != "rubric")
    elif flow == "custom_topics":
        payload = parse_json_object(raw_output)
        names.update(str(key) for key in payload)
        for section in payload.get("sections") or []:
            if isinstance(section, dict):
                names.update(str(key) for key in section)
    return names - set(server_assigned_fields(flow)) if flow in FLOW_CONTRACTS else names


#: Real fence delimiters only, anchored at the start of a line the way
#: :func:`app.services.prompt_blocks.untrusted_block` emits them. A bare
#: ``OPEN_TAG in prompt`` count is wrong: ``UNTRUSTED_CLAUSE`` legitimately
#: mentions ``<untrusted_input>`` in prose, which is not a fence.
_FENCE_OPENER = re.compile(rf"^{re.escape(OPEN_TAG)}(?:\s|>|$)", re.MULTILINE)
_FENCE_CLOSER = re.compile(rf"^{re.escape(CLOSE_TAG)}\s*$", re.MULTILINE)


def untrusted_containment_violations(prompt: str, hostile_values: Sequence[str]) -> list[str]:
    """Reasons a hostile input escaped (or corrupted) its fence.

    Two properties, both checkable without a model:

    1. Fence openers and closers are balanced, so no untrusted block can be left
       dangling and let following text read as trusted instructions.
    2. A hostile line that production's sanitiser *would* change never appears
       verbatim in the rendered prompt. Fence-token-only lines are exempt: a
       literal ``</untrusted_input>`` in the prompt is the legitimate closer,
       and property 1 already accounts for those.
    """
    violations: list[str] = []
    opens = len(_FENCE_OPENER.findall(prompt))
    closes = len(_FENCE_CLOSER.findall(prompt))
    if opens != closes:
        violations.append(f"unbalanced_untrusted_fence:opens={opens},closes={closes}")
    if hostile_values and opens == 0:
        violations.append("hostile_input_rendered_without_any_fence")
    for value in hostile_values:
        for raw_line in str(value or "").splitlines():
            line = raw_line.strip()
            if not line or _FENCE_TOKENS.fullmatch(line):
                continue
            if line in prompt and neutralise_fence(line) != line:
                violations.append("hostile_instruction_rendered_verbatim")
                break
    return violations


# ── per-flow case validation, all through production ───────────────────────


@dataclass
class CaseResult:
    case_id: str
    flow: str
    tags: list[str]
    valid: bool
    expected_valid: bool
    issues: list[str] = field(default_factory=list)
    item_count: int = 0
    duplicates: int = 0
    grounded: int = 0
    groundable: int = 0
    degraded: bool = False
    note_repaired: bool = False
    prompt: str = ""
    prompt_digest: str = ""
    prompt_violations: list[str] = field(default_factory=list)
    output_field_violations: list[str] = field(default_factory=list)
    containment_violations: list[str] = field(default_factory=list)
    digest_expected: str = ""


def _grounding_overlap(quote: str, source: str) -> bool:
    quote_tokens = _grounding_tokens(quote)
    source_tokens = _grounding_tokens(source)
    return bool(quote_tokens and source_tokens and (quote_tokens & source_tokens))


def _grounding_reference(case_input: dict[str, Any]) -> str:
    """The text a question is expected to be grounded in.

    Production grounds a question against the topic title and anchors lifted from
    the section (or the whole doc when there is no section), so the measurement
    uses the same material rather than an arbitrary excerpt.
    """
    return " ".join(
        str(case_input.get(key) or "")
        for key in ("topic_title", "section_title", "section_content", "doc_content")
    )


def _validate_questions_case(case_input: dict[str, Any], raw_output: str) -> tuple[bool, list[str], int, int, int, int]:
    items = _parse_questions_json(raw_output)
    if not items:
        return False, ["production_parser_returned_no_items"], 0, 0, 0, 0

    topic_id = str(case_input.get("topic_id", ""))
    topic_title = str(case_input.get("topic_title", ""))
    anchors = _grounding_anchor_phrases(
        topic_title=topic_title,
        section_title=case_input.get("section_title"),
        content=str(
            case_input.get("section_content")
            or case_input.get("doc_content", "")
        ),
    )
    issues: list[str] = []
    duplicates = 0
    grounded = 0
    seen: set[str] = set()
    reference = _grounding_reference(case_input)
    expected_count = int(case_input.get("count", 0) or 0)
    for item in items:
        ok, issue = _validate_question_item(
            dict(item),
            topic_id,
            case_input.get("difficulty"),
            case_input.get("level"),
            preferred_language=str(case_input.get("preferred_language", "")),
            requires_programming=bool(case_input.get("requires_programming", False)),
            topic_title=topic_title,
            grounding_anchors=anchors,
        )
        if not ok:
            issues.append(f"production_validator:{issue}")
            continue
        question_norm = normalise_question(str(item.get("question", "")))
        if question_norm in seen:
            duplicates += 1
        seen.add(question_norm)
        if _grounding_overlap(str(item.get("source_quote", "")), reference):
            grounded += 1
    if expected_count and len(items) != expected_count:
        issues.append(f"item_count_mismatch:expected={expected_count},got={len(items)}")
    return (not issues), issues, len(items), duplicates, grounded, len(items)


def _validate_quiz_case(case_input: dict[str, Any], raw_output: str) -> tuple[bool, list[str], int, int, int, int]:
    items = _parse_questions_json(raw_output)
    if not items:
        return False, ["production_parser_returned_no_items"], 0, 0, 0, 0

    topics = [dict(topic) for topic in case_input.get("topics") or []]
    allowed_topics = {str(topic.get("id", "")).strip() for topic in topics}
    allowed_types = set(case_input.get("question_types") or ["mcq", "true_false"])
    issues: list[str] = []
    duplicates = 0
    grounded = 0
    seen: set[str] = set()
    expected_count = int(case_input.get("count", 0) or 0)
    sources = " ".join(str(topic.get("content", "")) for topic in topics) + " " + " ".join(
        str(topic.get("title", "")) for topic in topics
    )
    for item in items:
        ok, issue = _validate_quiz_item(
            dict(item),
            allowed_topics,
            allowed_types,
            case_input.get("difficulty"),
            case_input.get("level"),
        )
        if not ok:
            issues.append(f"production_validator:{issue}")
            continue
        question_norm = normalise_question(str(item.get("question", "")))
        if question_norm in seen:
            duplicates += 1
        seen.add(question_norm)
        if _grounding_overlap(str(item.get("source_quote", "")), sources):
            grounded += 1
    if expected_count and len(items) != expected_count:
        issues.append(f"item_count_mismatch:expected={expected_count},got={len(items)}")
    return (not issues), issues, len(items), duplicates, grounded, len(items)


def _validate_interview_question_case(
    case_input: dict[str, Any], raw_output: str
) -> tuple[bool, list[str], int]:
    payload = parse_json_object(raw_output)
    if not payload:
        return False, ["production_parser_returned_no_object"], 0
    asked = [str(q) for q in (case_input.get("session") or {}).get("asked_questions") or []]
    ok, issue = InterviewGenerator._validate_question_payload(payload, asked)
    return ok, ([] if ok else [f"production_validator:{issue}"]), 1


def _validate_interview_eval_case(
    case_input: dict[str, Any], raw_output: str
) -> tuple[bool, list[str], int, bool]:
    """Score one evaluation the way production does.

    The ``follow_up_note_repaired`` flag comes from
    ``InterviewGenerator._normalise_eval_payload`` (Stage 1.4), so the repair
    rate is measured rather than asserted. An output production cannot use at
    all is the Stage 1.3 ``degraded`` path and is reported as such.
    """
    payload = parse_json_object(raw_output)
    if not payload:
        return False, ["degraded:production_parser_returned_no_object"], 1, False
    ok, issue = InterviewGenerator._validate_eval_payload(payload)
    if not ok:
        return False, [f"degraded:production_validator:{issue}"], 1, False
    normalised = _INTERVIEW_EVAL_NORMALISER._normalise_eval_payload(
        payload, session=dict(case_input.get("session") or {})
    )
    return True, [], 1, bool(normalised.get("follow_up_note_repaired"))


def _validate_chat_case(case_input: dict[str, Any], raw_output: str) -> tuple[bool, list[str], int]:
    reply = str(raw_output or "").strip()
    issues: list[str] = []
    if len(reply) < 20:
        issues.append("reply_too_short")
    if not reply:
        issues.append("empty_reply")
    for expected in case_input.get("expected_markers") or []:
        if str(expected).lower() not in reply.lower():
            issues.append(f"reply_missing_marker:{expected}")
    if "</untrusted_input>" in reply or OPEN_TAG in reply:
        issues.append("reply_leaks_untrusted_fence_delimiter")
    return (not issues), issues, 1


def _validate_custom_topic_case(case_input: dict[str, Any], raw_output: str) -> tuple[bool, list[str], int]:
    payload = parse_json_object(raw_output)
    if not payload:
        return False, ["production_parser_returned_no_object"], 0
    sections = payload.get("sections")
    issues: list[str] = []
    if not isinstance(sections, list) or not sections:
        return False, ["missing_sections"], 0
    expected_count = int(case_input.get("batch_size", 0) or 0)
    if expected_count and len(sections) != expected_count:
        issues.append(f"section_count_mismatch:expected={expected_count},got={len(sections)}")
    seen: set[str] = set()
    for section in sections:
        if not isinstance(section, dict):
            issues.append("invalid_section_item")
            continue
        heading = normalise_question(str(section.get("heading", "")))
        if not heading:
            issues.append("empty_section_heading")
            continue
        if heading in seen:
            issues.append("duplicate_section_heading")
        seen.add(heading)
    declared_total = int(case_input.get("expected_total_sections", 0) or 0)
    if declared_total and len(sections) < declared_total:
        issues.append(f"sections_below_declared_total:{len(sections)}<{declared_total}")
    return (not issues), issues, len(sections)


# ── fixture loading and prompt snapshots ───────────────────────────────────


def load_cases(fixture_dir: Path) -> list[CaseResult]:
    """Build every case by running the production validators over the fixtures."""
    results: list[CaseResult] = []
    snapshots = load_prompt_snapshots(fixture_dir)
    seen_ids: set[str] = set()

    for flow, filename in FLOW_FILES.items():
        for index, row in enumerate(_read_jsonl(fixture_dir / filename)):
            case_id = str(row.get("case_id") or f"{flow}.{index}")
            if case_id in seen_ids:
                raise ValueError(f"duplicate case_id {case_id!r} in {filename}")
            seen_ids.add(case_id)
            offending = FORBIDDEN_FIXTURE_KEYS.intersection(row)
            if offending:
                raise ValueError(
                    f"{filename}:{case_id} carries hand-typed metric field(s) "
                    f"{sorted(offending)}; retries must come from an instrumented run"
                )
            case_input = dict(row.get("input") or {})
            if not case_input:
                raise ValueError(f"{filename}:{case_id} has no input prompt inputs")

            raw_output = row.get("output")
            if isinstance(raw_output, str):
                raw_text = raw_output
            else:
                raw_text = json.dumps(raw_output, ensure_ascii=True)
            expected_valid = bool(row.get("expected", {}).get("valid", True))

            if flow == "questions":
                valid, issues, items, dups, grounded, groundable = _validate_questions_case(
                    case_input, raw_text
                )
                degraded = note_repaired = False
            elif flow == "quiz":
                valid, issues, items, dups, grounded, groundable = _validate_quiz_case(
                    case_input, raw_text
                )
                degraded = note_repaired = False
            elif flow == "interview_question":
                valid, issues, items = _validate_interview_question_case(case_input, raw_text)
                dups = grounded = groundable = 0
                degraded = not valid
                note_repaired = False
            elif flow == "interview":
                valid, issues, items, note_repaired = _validate_interview_eval_case(
                    case_input, raw_text
                )
                dups = grounded = groundable = 0
                degraded = not valid
            elif flow == "chat":
                valid, issues, items = _validate_chat_case(case_input, raw_text)
                dups = grounded = groundable = 0
                degraded = not valid
                note_repaired = False
            else:
                valid, issues, items = _validate_custom_topic_case(case_input, raw_text)
                dups = grounded = groundable = 0
                degraded = not valid
                note_repaired = False

            observed_valid = valid
            if not expected_valid and valid:
                # The fixture says this output must not validate. It did, so the
                # case is a *mismatch*, not a pass: a degraded path that quietly
                # started producing rubrics is exactly the regression to catch.
                issues = list(issues) + ["expected_degraded_but_validated"]

            prompt = rebuild_prompt(flow, case_input)
            result = CaseResult(
                case_id=case_id,
                flow=flow,
                tags=[str(tag) for tag in row.get("tags") or []],
                valid=observed_valid,
                expected_valid=expected_valid,
                issues=sorted(set(issues)),
                item_count=items,
                duplicates=dups,
                grounded=grounded,
                groundable=groundable,
                degraded=degraded,
                note_repaired=note_repaired,
                prompt=prompt,
                prompt_digest=_digest(prompt),
                prompt_violations=prompt_contract_violations(flow, prompt),
                output_field_violations=prompt_output_field_violations(
                    flow, prompt, raw_text, observed_valid
                ),
                containment_violations=untrusted_containment_violations(
                    prompt, [str(v) for v in case_input.get("hostile_inputs") or []]
                ),
                digest_expected=str(snapshots.get(case_id, "")),
            )
            results.append(result)
    return results


def snapshot_path(fixture_dir: Path) -> Path:
    return fixture_dir / "prompt_snapshots.json"


def load_prompt_snapshots(fixture_dir: Path) -> dict[str, str]:
    path = snapshot_path(fixture_dir)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("cases", {}) if isinstance(data, dict) else {}
    return {str(k): str(v) for k, v in entries.items()} if isinstance(entries, dict) else {}


def write_prompt_snapshots(results: Sequence[CaseResult], fixture_dir: Path) -> Path:
    """Re-baseline the golden prompt digests.

    Only ever called explicitly (``--blame-prompts``): it is the documented
    escape hatch for an intentional prompt edit, and it makes that edit visible
    in review rather than silently accepted.
    """
    path = snapshot_path(fixture_dir)
    payload = {
        "_comment": (
            "Golden sha256 of each production input prompt rebuilt from a fixture's "
            "input. Regenerate with `python scripts/eval_agent_quality.py --blame-prompts` "
            "after an intentional prompt change."
        ),
        "cases": {result.case_id: result.prompt_digest for result in sorted(results, key=lambda r: r.case_id)},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ── fixture matrix coverage ───────────────────────────────────────────────

#: Dimensions the fixture matrix must cover. A new language in
#: ``PROBLEM_SOLVING_LANGUAGE_OPTIONS`` (Stage 1.10) or a new interview type
#: fails this gate until a fixture exists for it, so "expand the matrix" is an
#: enforced requirement rather than a note in a plan.
MATRIX_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "level": ("junior", "mid", "senior"),
    "interview_type": ("coding", "behavioral", "mixed"),
    "problem_solving_language": tuple(PROBLEM_SOLVING_LANGUAGE_OPTIONS),
}


def matrix_coverage(cases: Sequence[CaseResult]) -> dict[str, Any]:
    covered: dict[str, list[str]] = {}
    for dimension, expected in MATRIX_DIMENSIONS.items():
        seen = {tag for case in cases for tag in case.tags if tag in expected}
        covered[dimension] = sorted(expected)
        covered[f"{dimension}_seen"] = sorted(seen)
        covered[f"{dimension}_missing"] = sorted(set(expected) - seen)
        covered[f"{dimension}_rate"] = _rate(len(seen), len(expected))
    return covered


def _matrix_failures(coverage: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "gate": f"matrix_{dimension}_coverage",
            "observed": coverage[f"{dimension}_rate"],
            "detail": f"no fixture covers {coverage[f'{dimension}_missing']}",
        }
        for dimension in MATRIX_DIMENSIONS
        if coverage[f"{dimension}_missing"]
    ]


# ── retry distribution from instrumented runs ──────────────────────────────


@dataclass
class RetryDistribution:
    available: bool
    verified: bool
    source: str
    calls: int = 0
    groups: int = 0
    retried_groups: int = 0
    retry_rate: float = 0.0
    mean_attempts: float = 0.0
    max_attempts: int = 0
    attempts_histogram: dict[str, int] = field(default_factory=dict)
    error_codes: dict[str, int] = field(default_factory=dict)
    finish_reasons: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0
    truncated_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "verified": self.verified,
            "source": self.source,
            "calls": self.calls,
            "attempt_groups": self.groups,
            "retried_groups": self.retried_groups,
            "retry_rate": self.retry_rate,
            "mean_attempts": self.mean_attempts,
            "max_attempts": self.max_attempts,
            "attempts_histogram": self.attempts_histogram,
            "error_codes": self.error_codes,
            "finish_reasons": self.finish_reasons,
            "total_tokens": self.total_tokens,
            "truncated_calls": self.truncated_calls,
        }


def retry_distribution(fixture_dir: Path) -> RetryDistribution:
    """Derive retries by grouping recorded ``LLMClient`` call records.

    Each line of the run store is **one** ``LLMClient.completion`` call, carrying
    the ``usage`` / ``finish_reason`` / ``error_code`` metadata that
    ``llm_client.completion`` already returns. Attempts are the number of call
    records sharing one ``group_id`` (one attempt cycle for one requested
    output). A file cannot simply declare "I used 2 retries".

    Records flagged ``synthetic`` exercise the arithmetic without being
    evidence, so they are counted but never satisfy the gate.
    """
    real = _read_jsonl(fixture_dir / "runs" / RUN_STORE_NAME)
    synthetic = _read_jsonl(fixture_dir / "runs" / SYNTHETIC_RUN_STORE_NAME)
    records = real + synthetic
    if not records:
        return RetryDistribution(
            available=False,
            verified=False,
            source=None,  # type: ignore[arg-type]
        )

    for record in records:
        offending = FORBIDDEN_FIXTURE_KEYS.intersection(record)
        if offending:
            raise ValueError(
                f"run store carries hand-typed metric field(s) {sorted(offending)}; "
                "attempts are counted from grouped call records only"
            )

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("group_id") or record.get("run_id") or "")
        if not key:
            raise ValueError("every run-store record needs a group_id to be countable")
        groups.setdefault(key, []).append(record)

    histogram = Counter(len(calls) for calls in groups.values())
    attempts = sorted(histogram.elements())
    error_codes = Counter(
        str(record.get("error_code") or "").strip() or "none"
        for record in records
    )
    finish_reasons = Counter(
        str(record.get("finish_reason") or "").strip() or "none" for record in records
    )
    total_tokens = 0
    for record in records:
        usage = record.get("usage")
        if isinstance(usage, dict):
            total_tokens += int(usage.get("total_tokens") or 0)
    retried = sum(1 for calls in groups.values() if len(calls) > 1)
    return RetryDistribution(
        available=True,
        verified=bool(real),
        source=f"evals/runs/{RUN_STORE_NAME}"
        + (" (+ synthetic)" if synthetic else ""),
        calls=len(records),
        groups=len(groups),
        retried_groups=retried,
        retry_rate=round((retried / len(groups)) * 100, 2) if groups else 0.0,
        mean_attempts=round(sum(attempts) / len(attempts), 2) if attempts else 0.0,
        max_attempts=max(attempts) if attempts else 0,
        attempts_histogram={str(k): v for k, v in sorted(histogram.items())},
        error_codes=dict(error_codes.most_common()),
        finish_reasons=dict(finish_reasons.most_common()),
        total_tokens=total_tokens,
        truncated_calls=int(
            sum(1 for r in records if str(r.get("finish_reason") or "").lower() == "length")
        ),
    )


# ── report assembly ────────────────────────────────────────────────────────


def _rate(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100, 2) if denominator else 0.0


def evaluate_gates(
    metrics: dict[str, Any], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    def check(name: str, observed: Any, ok: bool, detail: str) -> None:
        if not ok:
            failures.append({"gate": name, "observed": observed, "detail": detail})

    check(
        "schema_valid_rate",
        metrics["schema_valid_rate"],
        metrics["schema_valid_rate"] >= thresholds["schema_valid_rate_min"],
        f"must be >= {thresholds['schema_valid_rate_min']}%",
    )
    check(
        "prompt_contract_rate",
        metrics["prompt_contract_rate"],
        metrics["prompt_contract_rate"] >= thresholds["prompt_contract_rate_min"],
        "every rebuilt prompt must name each field its production validator requires",
    )
    check(
        "prompt_output_field_rate",
        metrics["prompt_output_field_rate"],
        metrics["prompt_output_field_rate"] >= thresholds["prompt_output_field_rate_min"],
        "every field the model emitted must be named by the prompt that asked for it",
    )
    check(
        "untrusted_containment_rate",
        metrics["untrusted_containment_rate"],
        metrics["untrusted_containment_rate"] >= thresholds["untrusted_containment_rate_min"],
        "hostile inputs must stay inside a balanced <untrusted_input> fence",
    )
    check(
        "prompt_digest_match_rate",
        metrics["prompt_digest_match_rate"],
        metrics["prompt_digest_match_rate"] >= thresholds["prompt_digest_match_rate_min"],
        "an unbaselined prompt edit; run --blame-prompts only when intended",
    )
    check(
        "duplicate_rate",
        metrics["duplicate_rate"],
        metrics["duplicate_rate"] <= thresholds["duplicate_rate_max"],
        f"must be <= {thresholds['duplicate_rate_max']}%",
    )
    check(
        "grounding_fidelity",
        metrics["grounding_fidelity"],
        metrics["grounding_fidelity"] >= thresholds["grounding_fidelity_min"],
        f"must be >= {thresholds['grounding_fidelity_min']}%",
    )
    check(
        "repair_rate",
        metrics["repair_rate"],
        metrics["repair_rate"] <= thresholds["repair_rate_max"],
        f"must be <= {thresholds['repair_rate_max']}% (Stage 1.4 follow_up_note_repaired)",
    )
    check(
        "degraded_eval_rate",
        metrics["degraded_eval_rate"],
        metrics["degraded_eval_rate"] <= thresholds["degraded_eval_rate_max"],
        f"must be <= {thresholds['degraded_eval_rate_max']}% (Stage 1.3 degraded flag)",
    )
    retry = metrics["retry_distribution"]
    if retry["available"] and retry["verified"]:
        check(
            "retry_rate",
            retry["retry_rate"],
            retry["retry_rate"] <= thresholds["retry_rate_max"],
            f"must be <= {thresholds['retry_rate_max']}%",
        )
        check(
            "max_attempts",
            retry["max_attempts"],
            retry["max_attempts"] <= thresholds["max_attempts_max"],
            f"must be <= {thresholds['max_attempts_max']} per attempt cycle",
        )
    return failures


def unverified_gates(metrics: dict[str, Any]) -> list[str]:
    retry = metrics["retry_distribution"]
    if retry["available"] and not retry["verified"]:
        return ["retry_rate", "max_attempts"]
    if not retry["available"]:
        return ["retry_rate", "max_attempts"]
    return []


def run_eval(
    fixture_dir: Path,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Deterministic tier: production validators over checked-in fixtures."""
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    cases = load_cases(fixture_dir)
    retries = retry_distribution(fixture_dir)

    total_cases = len(cases)
    matched_cases = sum(1 for case in cases if case.valid == case.expected_valid)
    valid_cases = sum(1 for case in cases if case.valid)
    expected_invalid = sum(1 for case in cases if not case.expected_valid)
    total_items = sum(case.item_count for case in cases)
    total_duplicates = sum(case.duplicates for case in cases)
    grounded_items = sum(case.grounded for case in cases)
    groundable_items = sum(case.groundable for case in cases)

    prompt_cases = [case for case in cases if case.prompt]
    prompt_contract_ok = sum(1 for case in prompt_cases if not case.prompt_violations)
    output_field_ok = sum(
        1
        for case in prompt_cases
        if not case.output_field_violations and case.valid == case.expected_valid
    )
    containment_ok = sum(1 for case in prompt_cases if not case.containment_violations)
    digest_baselined = [case for case in prompt_cases if case.digest_expected]
    digest_ok = sum(
        1 for case in digest_baselined if case.digest_expected == case.prompt_digest
    )

    eval_cases = [case for case in cases if case.flow == "interview"]
    repaired = sum(1 for case in eval_cases if case.note_repaired)
    degraded = sum(1 for case in eval_cases if case.degraded)

    summary: dict[str, Any] = {
        "total_cases": total_cases,
        # Every case counts, including the ones that are *supposed* to degrade:
        # the rate is the share whose observed validity matched its declaration,
        # so both "a good output broke" and "a degraded path stopped degrading"
        # move it down.
        "schema_valid_rate": _rate(matched_cases, total_cases),
        "schema_valid_cases": valid_cases,
        "schema_expected_invalid_cases": expected_invalid,
        "prompt_contract_rate": _rate(prompt_contract_ok, len(prompt_cases)),
        "prompt_output_field_rate": _rate(output_field_ok, len(prompt_cases)),
        "untrusted_containment_rate": _rate(containment_ok, len(prompt_cases)),
        "prompt_digest_match_rate": _rate(digest_ok, len(digest_baselined)),
        "prompt_digest_unbaselined": len(prompt_cases) - len(digest_baselined),
        "duplicate_rate": _rate(total_duplicates, total_items),
        "grounding_fidelity": _rate(grounded_items, groundable_items),
        "repair_rate": _rate(repaired, len(eval_cases)),
        "degraded_eval_rate": _rate(degraded, len(eval_cases)),
        "retry_distribution": retries.as_dict(),
    }

    per_flow: dict[str, dict[str, Any]] = {}
    for flow in FLOW_FILES:
        flow_cases = [case for case in cases if case.flow == flow]
        flow_valid = sum(1 for case in flow_cases if case.valid)
        flow_prompts = [case for case in flow_cases if case.prompt]
        per_flow[flow] = {
            "total": len(flow_cases),
            "valid": flow_valid,
            "expected_invalid": sum(1 for case in flow_cases if not case.expected_valid),
            "schema_valid_rate": _rate(
                sum(1 for case in flow_cases if case.valid == case.expected_valid),
                len(flow_cases),
            ),
            "prompt_contract_rate": _rate(
                sum(1 for case in flow_prompts if not case.prompt_violations),
                len(flow_prompts),
            ),
            "prompt_output_field_rate": _rate(
                sum(
                    1
                    for case in flow_prompts
                    if not case.output_field_violations
                    and case.valid == case.expected_valid
                ),
                len(flow_prompts),
            ),
            "untrusted_containment_rate": _rate(
                sum(1 for case in flow_prompts if not case.containment_violations),
                len(flow_prompts),
            ),
            "degraded_cases": sum(1 for case in flow_cases if case.degraded),
            "repaired_cases": sum(1 for case in flow_cases if case.note_repaired),
        }

    metrics = dict(summary)
    metrics["retry_distribution"] = retries.as_dict()
    coverage = matrix_coverage(cases)
    failures = evaluate_gates(metrics, thresholds) + _matrix_failures(coverage)

    return {
        "tier": "deterministic",
        "fixture_dir": str(fixture_dir),
        "summary": summary,
        "flows": per_flow,
        "matrix_coverage": coverage,
        "contracts": {
            flow: sorted(contract.fields) for flow, contract in FLOW_CONTRACTS.items()
        },
        "thresholds": thresholds,
        "gate_failures": failures,
        "unverified_gates": unverified_gates(metrics),
        "failing_cases": [
            {
                "case_id": case.case_id,
                "flow": case.flow,
                "expected_valid": case.expected_valid,
                "observed_valid": case.valid,
                "issues": case.issues,
                "prompt_violations": case.prompt_violations,
                "output_field_violations": case.output_field_violations,
                "containment_violations": case.containment_violations,
                "prompt_digest_drift": bool(
                    case.digest_expected and case.digest_expected != case.prompt_digest
                ),
            }
            for case in cases
            if case.valid != case.expected_valid
            or case.prompt_violations
            or case.output_field_violations
            or case.containment_violations
            or (case.digest_expected and case.digest_expected != case.prompt_digest)
        ],
    }


# ── live tier: regenerate, record telemetry, diff the golden store ─────────


class LLMCallRecorder:
    """An ``LLMClient`` stand-in that records one row per real call.

    This is where retry counts come from. The recorder is never told a retry
    count; it records the metadata ``LLMClient.completion`` already returns and
    lets :func:`retry_distribution` group the rows afterwards.

    It mirrors ``LLMClient.completion``'s signature so the production generators
    can hold it in place of a client unchanged.
    """

    def __init__(self, client: Any, sink: list[dict[str, Any]], *, run_id: str) -> None:
        self._client = client
        self._sink = sink
        self._run_id = run_id
        self._groups: Counter[str] = Counter()

    def completion(
        self,
        prompt: str,
        llm_config: Any = None,
        user_identity: dict[str, Any] | None = None,
        task: str | None = None,
        *,
        system: str = "",
        structured: bool = False,
        max_tokens_cap: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        eval_group_id: str = "",
    ) -> Any:
        group_id = str(eval_group_id or self._run_id)
        result = self._client.completion(
            prompt,
            llm_config,
            user_identity,
            task,
            system=system,
            structured=structured,
            max_tokens_cap=max_tokens_cap,
            tools=tools,
            tool_choice=tool_choice,
        )
        self._record(group_id, {"task": task or ""}, result)
        return result

    def _record(self, group_id: str, kwargs: dict[str, Any], result: Any) -> None:
        self._groups[group_id] += 1
        metadata = result.get("metadata") if isinstance(result, dict) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        error_code = str(metadata.get("error_code") or "")
        if not error_code and isinstance(result, dict):
            error_code = str(result.get("error_code") or "")
        self._sink.append(
            {
                "run_id": self._run_id,
                "group_id": group_id,
                "sequence": self._groups[group_id],
                "flow": str(kwargs.get("task") or ""),
                "provider": str(metadata.get("provider") or ""),
                "model": str(metadata.get("model") or ""),
                "finish_reason": str(result.get("finish_reason") or "")
                if isinstance(result, dict)
                else "",
                "error_code": error_code,
                "usage": result.get("usage") if isinstance(result, dict) else {},
            }
        )

    def calls(self) -> list[dict[str, Any]]:
        return list(self._sink)


def _golden_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("case_id")): row for row in _read_jsonl(path)}


def diff_golden(fixture_dir: Path, golden_dir: Path) -> dict[str, Any]:
    """Compare checked-in golden outputs with the current fixture outputs."""
    diffs: dict[str, Any] = {}
    for flow, filename in FLOW_FILES.items():
        golden_path = golden_dir / filename
        if not golden_path.exists():
            diffs[flow] = {"status": "no_golden_store"}
            continue
        golden = _golden_rows(golden_path)
        current = {
            str(row.get("case_id")): row for row in _read_jsonl(fixture_dir / filename)
        }
        added = sorted(set(current) - set(golden))
        removed = sorted(set(golden) - set(current))
        changed = []
        for case_id in sorted(set(golden) & set(current)):
            before = json.dumps(golden[case_id].get("output"), indent=2, sort_keys=True)
            after = json.dumps(current[case_id].get("output"), indent=2, sort_keys=True)
            if before != after:
                changed.append(
                    {
                        "case_id": case_id,
                        "diff": "\n".join(
                            difflib.unified_diff(
                                before.splitlines(),
                                after.splitlines(),
                                fromfile="golden",
                                tofile="current",
                                lineterm="",
                            )
                        )[:4000],
                    }
                )
        diffs[flow] = {
            "status": "ok",
            "added": added,
            "removed": removed,
            "changed": changed,
            "drift": bool(added or removed or changed),
        }
    return diffs


#: Fixture ``input`` keys that are not request parameters. ``hostile_inputs``
#: describes the fixture, not the request.
_NON_REQUEST_INPUT_KEYS = {"hostile_inputs", "recorded_prompt", "expected_markers"}

#: Fixture input keys whose production parameter has a different name.
_LIVE_INPUT_ALIASES = {"quiz": {"topics": "topics_content"}}


def _live_kwargs(flow: str, case_input: dict[str, Any], parameters: set[str]) -> dict[str, Any]:
    """Map a fixture's ``input`` onto one generator's keyword parameters."""
    aliases = _LIVE_INPUT_ALIASES.get(flow, {})
    kwargs: dict[str, Any] = {}
    for key, value in case_input.items():
        if key in _NON_REQUEST_INPUT_KEYS:
            continue
        name = aliases.get(key, key)
        if name not in parameters or value is None:
            continue
        kwargs[name] = value
    return kwargs


async def _live_regenerate(
    cases: Sequence[CaseResult],
    fixture_dir: Path,
    golden_dir: Path,
    flows: Sequence[str],
) -> dict[str, Any]:
    """Replay fixture inputs through the real generators and record telemetry.

    Requires provider credentials; never runs in the deterministic CI tier. A
    per-case failure is recorded in ``errors`` and the run continues, so one
    provider outage does not cost the whole report.
    """
    import inspect

    from app.config import get_settings
    from app.services.doc_parser import DocParser
    from app.services.llm_client import LLMClient
    from app.services.question_generator import QuestionGenerator

    run_id = "live-" + hashlib.sha256(
        ",".join(sorted(case.case_id for case in cases)).encode()
    ).hexdigest()[:12]
    sink: list[dict[str, Any]] = []
    recorder = LLMCallRecorder(LLMClient(), sink, run_id=run_id)
    parser = DocParser()
    questions = QuestionGenerator(recorder)
    interview = InterviewGenerator(recorder, parser)

    question_params = set(inspect.signature(QuestionGenerator.generate_v2).parameters)
    quiz_params = set(inspect.signature(QuestionGenerator.generate_quiz_v2).parameters)

    errors: list[dict[str, str]] = []
    fresh: dict[str, dict[str, Any]] = {flow: {} for flow in FLOW_FILES}

    for case in cases:
        if case.flow not in flows:
            continue
        case_input = _case_input_for(fixture_dir, case)
        try:
            if case.flow == "questions":
                result = await questions.generate_v2(
                    **_live_kwargs("questions", case_input, question_params)
                )
                output = json.dumps(
                    [item.model_dump() for item in result.questions], ensure_ascii=True
                )
            elif case.flow == "quiz":
                result = await questions.generate_quiz_v2(
                    **_live_kwargs("quiz", case_input, quiz_params)
                )
                output = json.dumps(
                    [item.model_dump() for item in result.questions], ensure_ascii=True
                )
            elif case.flow == "interview_question":
                output = json.dumps(
                    await interview.generate_question(
                        session=dict(case_input.get("session") or {}),
                        turns=list(case_input.get("turns") or []),
                    ),
                    ensure_ascii=True,
                )
            elif case.flow == "interview":
                output = json.dumps(
                    await interview.evaluate_answer(
                        session=dict(case_input.get("session") or {}),
                        question=str(case_input.get("question", "")),
                        user_answer=str(case_input.get("answer", "")),
                        turn_index=int(case_input.get("turn_index", 0) or 0),
                    ),
                    ensure_ascii=True,
                )
            else:
                errors.append(
                    {
                        "case_id": case.case_id,
                        "error": f"no live driver for flow {case.flow!r}",
                    }
                )
                continue
        except Exception as exc:  # pragma: no cover - live tier only
            errors.append({"case_id": case.case_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        fresh[case.flow][case.case_id] = {
            "case_id": case.case_id,
            "output": output,
            "captured_at_run": run_id,
        }

    _write_jsonl(fixture_dir / "runs" / RUN_STORE_NAME, sink)
    return {
        "run_id": run_id,
        "provider_configured": bool(getattr(get_settings(), "groq_api_key", "")),
        "calls_recorded": len(sink),
        "errors": errors,
        "fresh": {flow: rows for flow, rows in fresh.items() if rows},
        "golden_dir": str(golden_dir),
    }


def _case_input_for(fixture_dir: Path, case: CaseResult) -> dict[str, Any]:
    filename = FLOW_FILES[case.flow]
    for row in _read_jsonl(fixture_dir / filename):
        if str(row.get("case_id")) == case.case_id:
            return dict(row.get("input") or {})
    return {}


# ── CLI ────────────────────────────────────────────────────────────────────


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Agent Quality Eval")
    print(f"- tier: {report.get('tier', 'deterministic')}")
    print(f"- total_cases: {summary['total_cases']}")
    print(f"- schema_valid_rate: {summary['schema_valid_rate']}%")
    print(f"- prompt_contract_rate: {summary['prompt_contract_rate']}%")
    print(f"- prompt_output_field_rate: {summary['prompt_output_field_rate']}%")
    print(f"- untrusted_containment_rate: {summary['untrusted_containment_rate']}%")
    print(f"- prompt_digest_match_rate: {summary['prompt_digest_match_rate']}%")
    print(f"- duplicate_rate: {summary['duplicate_rate']}%")
    print(f"- grounding_fidelity: {summary['grounding_fidelity']}%")
    print(f"- repair_rate: {summary['repair_rate']}%")
    print(f"- degraded_eval_rate: {summary['degraded_eval_rate']}%")
    retry = summary["retry_distribution"]
    if retry["available"]:
        provenance = "verified instrumented run" if retry["verified"] else "synthetic only (unverified)"
        print(
            f"- retry_rate: {retry['retry_rate']}% "
            f"(max_attempts={retry['max_attempts']}, calls={retry['calls']}, {provenance})"
        )
    else:
        print("- retry_rate: unavailable (no instrumented run recorded yet)")
    for flow, data in report["flows"].items():
        print(f"- {flow}: {data['valid']}/{data['total']} ({data['schema_valid_rate']}%)")
    coverage = report.get("matrix_coverage") or {}
    for dimension in sorted(MATRIX_DIMENSIONS):
        print(f"- matrix {dimension}: {coverage.get(f'{dimension}_rate')}% covered")
    if report.get("gate_failures"):
        print("\nGATE FAILURES")
        for failure in report["gate_failures"]:
            print(f"- {failure['gate']}: observed {failure['observed']} ({failure['detail']})")
    if report.get("unverified_gates"):
        print(f"\nUNVERIFIED GATES (need an instrumented run): {', '.join(report['unverified_gates'])}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline eval for agent quality metrics.")
    parser.add_argument(
        "--fixture-dir",
        default=str(BACKEND_ROOT / "evals"),
        help="Directory with JSONL eval fixtures.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    parser.add_argument(
        "--blame-prompts",
        action="store_true",
        help="Re-baseline the golden prompt digests and exit.",
    )
    parser.add_argument(
        "--golden-dir",
        default=None,
        help="Golden-output store for the live tier (default: <fixture-dir>/golden).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live tier: replay inputs through the real generators (costs money, needs keys).",
    )
    parser.add_argument(
        "--live-flow",
        default=",".join(FLOW_FILES),
        help="Comma-separated flows the live tier should regenerate.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    fixture_dir = Path(args.fixture_dir)
    golden_dir = Path(args.golden_dir) if args.golden_dir else fixture_dir / "golden"

    if args.blame_prompts:
        path = write_prompt_snapshots(load_cases(fixture_dir), fixture_dir)
        print(f"Re-baselined prompt digests: {path}")
        return 0

    if args.live:
        cases = load_cases(fixture_dir)
        live = asyncio.run(
            _live_regenerate(cases, fixture_dir, golden_dir, [f for f in args.live_flow.split(",") if f])
        )
        report = run_eval(fixture_dir)
        report["tier"] = "live"
        report["live"] = live
        report["golden_diff"] = diff_golden(fixture_dir, golden_dir)
        if args.json:
            print(json.dumps(report, ensure_ascii=True, indent=2))
        else:
            _print_report(report)
            print("\nGOLDEN DIFF")
            print(json.dumps(report["golden_diff"], ensure_ascii=True, indent=2))
        return 1 if report["gate_failures"] else 0

    report = run_eval(fixture_dir)
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        _print_report(report)
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
