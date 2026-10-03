"""``submit_rubric``: the interview evaluator's forced output tool.

Why a tool instead of ``response_format``
-----------------------------------------
``response_format={"type": "json_object"}`` only asks a model for *a* JSON
object; it does not enforce which keys, which ranges, or which array bounds
exist. The evaluator's rubric has six required keys, two bounded arrays, and a
0-5 vs 0-100 range split, and today every one of those constraints is discovered
by re-parsing, re-validating and re-prompting (``InterviewGenerator.
_validate_eval_payload`` → retry). A forced tool call moves the contract into
the schema the provider validates, so a malformed rubric is rejected by the
provider rather than costing a round trip.

Ready-but-not-wired
-------------------
This module is self-contained on purpose. ``interview_generator`` already
imports ``mcp_gateway`` which imports this package, so importing
``interview_generator`` here would create an import cycle. The payload contract
is therefore duplicated as :data:`RUBRIC_KEYS` / :data:`RUBRIC_RANGES` instead of
imported — the same values the evaluator's prompt block already declares. The
integration points that remain to be applied in
``app/services/interview_generator.py`` are listed in the plan's Stage 4.2.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.services.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

SUBMIT_RUBRIC_TOOL_NAME = "submit_rubric"

#: ``tool_choice`` payload that forces the single tool, so the model has no way
#: to answer in prose instead.
SUBMIT_RUBRIC_TOOL_CHOICE: dict[str, Any] = {
    "type": "function",
    "function": {"name": SUBMIT_RUBRIC_TOOL_NAME},
}

#: Rubric keys and their inclusive (min, max) bounds, matching the schema block
#: in ``InterviewGenerator._evaluate_prompt``.
RUBRIC_RANGES: dict[str, tuple[int, int]] = {
    "technical_accuracy": (0, 5),
    "reasoning_depth": (0, 5),
    "communication_clarity": (0, 5),
    "completeness": (0, 5),
    "confidence_signal": (0, 5),
    "overall": (0, 100),
}
RUBRIC_KEYS: tuple[str, ...] = tuple(RUBRIC_RANGES)

LIST_KEYS: tuple[str, ...] = ("strengths", "improvements")
LIST_MIN_ITEMS = 1
LIST_MAX_ITEMS = 4

#: Keys the evaluator reads at the top level. Anything else is dropped, so a
#: verbose model cannot smuggle extra structure into the payload. The six
#: rubric dimensions live inside ``rubric``, not at the top level.
ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"rubric", *LIST_KEYS, "follow_up_note"})


class RubricSubmissionError(ValueError):
    """The model submitted a payload the rubric contract does not allow."""


def _score_property(key: str) -> dict[str, Any]:
    low, high = RUBRIC_RANGES[key]
    return {
        "type": "integer",
        "minimum": low,
        "maximum": high,
        "description": f"0-{high} score. Only use {low} when there is no evidence either way.",
    }


def submit_rubric_parameters() -> dict[str, Any]:
    """JSON Schema for the rubric payload. Rebuilt per call so callers cannot
    mutate the shared definition."""
    return {
        "type": "object",
        "properties": {
            "rubric": {
                "type": "object",
                "properties": {key: _score_property(key) for key in RUBRIC_KEYS},
                "required": list(RUBRIC_KEYS),
                "additionalProperties": False,
                "description": "Every key is required. Do not omit a dimension you cannot judge.",
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": LIST_MIN_ITEMS,
                "maxItems": LIST_MAX_ITEMS,
                "description": "1-4 short, concrete strengths.",
            },
            "improvements": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": LIST_MIN_ITEMS,
                "maxItems": LIST_MAX_ITEMS,
                "description": "1-4 short, actionable improvements.",
            },
            "follow_up_note": {
                "type": "string",
                "description": (
                    "Markdown coaching note with these sections in order: "
                    "`### What strong interviewers wanted to hear`, "
                    "`### What to improve next`, `### Stronger sample answer`."
                ),
            },
        },
        "required": ["rubric", "strengths", "improvements", "follow_up_note"],
        "additionalProperties": False,
    }


def submit_rubric_tool(*, enabled: bool = True, max_result_chars: int = 4000) -> ToolSpec:
    """The ``submit_rubric`` tool spec.

    The handler is a validator, not a remote call: it returns the accepted
    payload as canonical JSON, so a caller that wires this into
    ``evaluate_answer`` can pass ``result.content`` straight into
    ``_normalise_eval_payload``.
    """

    async def submit_rubric(arguments: dict[str, Any]) -> str:
        payload = validate_rubric_submission(arguments)
        return json.dumps(payload, ensure_ascii=False)

    return ToolSpec(
        name=SUBMIT_RUBRIC_TOOL_NAME,
        description=(
            "Submit the scored rubric for this interview turn. Call this exactly "
            "once. This is the only way to return a result; do not answer in prose."
        ),
        parameters=submit_rubric_parameters(),
        handler=submit_rubric,
        enabled=bool(enabled),
        timeout_seconds=5.0,
        metadata={"flow": "interview", "mode": "forced_structured_output"},
    )


def _coerce_score(value: Any, key: str) -> int:
    low, high = RUBRIC_RANGES[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RubricSubmissionError(f"{key} must be a number, got {type(value).__name__}")
    if isinstance(value, float) and not value.is_integer():
        raise RubricSubmissionError(f"{key} must be a whole number, got {value}")
    score = int(value)
    if score < low or score > high:
        raise RubricSubmissionError(f"{key}={score} is outside the allowed range {low}-{high}")
    return score


def _coerce_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list):
        raise RubricSubmissionError(f"{key} must be an array, got {type(value).__name__}")
    items = [str(item).strip() for item in value if str(item).strip()]
    if len(items) < LIST_MIN_ITEMS:
        raise RubricSubmissionError(f"{key} needs at least {LIST_MIN_ITEMS} non-empty string")
    if len(items) > LIST_MAX_ITEMS:
        raise RubricSubmissionError(f"{key} allows at most {LIST_MAX_ITEMS} items, got {len(items)}")
    return items


def validate_rubric_submission(arguments: Any) -> dict[str, Any]:
    """Validate a rubric payload and return it in canonical form.

    Raises :class:`RubricSubmissionError` with a specific message so a caller
    can feed the reason back to the model as a correction instead of falling
    through to a fabricated score.
    """
    if not isinstance(arguments, dict):
        raise RubricSubmissionError(
            f"rubric payload must be an object, got {type(arguments).__name__}"
        )

    unknown = sorted(set(arguments) - ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise RubricSubmissionError(f"unexpected keys: {unknown}")

    rubric = arguments.get("rubric")
    if not isinstance(rubric, dict):
        raise RubricSubmissionError("missing_rubric: 'rubric' must be an object")
    missing = [key for key in RUBRIC_KEYS if key not in rubric]
    if missing:
        raise RubricSubmissionError(f"missing rubric keys: {missing}")
    unknown_rubric = sorted(set(rubric) - set(RUBRIC_KEYS))
    if unknown_rubric:
        raise RubricSubmissionError(f"unexpected rubric keys: {unknown_rubric}")

    note = arguments.get("follow_up_note")
    if not isinstance(note, str) or not note.strip():
        raise RubricSubmissionError("missing follow_up_note: a markdown coaching note is required")

    return {
        "rubric": {key: _coerce_score(rubric[key], key) for key in RUBRIC_KEYS},
        "strengths": _coerce_list(arguments.get("strengths"), "strengths"),
        "improvements": _coerce_list(arguments.get("improvements"), "improvements"),
        "follow_up_note": note,
    }
