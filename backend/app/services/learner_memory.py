"""LLM-visible learner state for generation prompts.

The generation prompts used to see only the per-topic quiz progress
note, so weak areas, objective accuracy, recurring error patterns, and
interview weaknesses were computed and stored but never reached the
model. This module gathers that learner state into ONE bounded, fenced
block a generation flow can interpolate.

Failure tolerance is the contract, not a nicety: every accessor is
duck-typed and guarded, so a store that is absent, lacks the accessor,
or raises degrades to a *skipped section* rather than a failed
generation. No user id, no data, or an exhausted budget all mean "no
block at all", so a prompt built without learner state stays
byte-identical to one built with an empty learner.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.services.prompt_blocks import section as _prompt_section
from app.services.prompt_blocks import untrusted_block

logger = logging.getLogger(__name__)

#: Default ceiling on the fenced block's inner content. The block rides
#: along on every generation prompt, so it stays a fixed-size addition.
DEFAULT_MAX_CHARS = 1200

#: Label on the single fence this module renders. The content is
#: server-side derived (never a raw client field), but it is still
#: fenced because error-pattern explanations are model-generated from
#: learner answers.
LEARNER_STATE_LABEL = "Learner state (server-side, trusted context)"

#: Per-section item caps. They bound a single section so the priority
#: walk below can stop on the budget instead of truncating mid-section.
_MAX_WEAK_AREAS = 4
_MAX_OBJECTIVES = 3
_MAX_ERROR_PATTERNS = 4
_MAX_INTERVIEW_DIMENSIONS = 3

#: How much of an error-pattern explanation is worth prompt space.
_EXPLANATION_CHARS = 140

#: Interview rubric dimensions, weakest first. ``overall`` is a composite
#: of the rest, so it is never listed as its own weakness.
_INTERVIEW_DIMENSIONS = (
    ("technical_accuracy", "technical accuracy"),
    ("reasoning_depth", "reasoning depth"),
    ("communication_clarity", "communication clarity"),
    ("completeness", "completeness"),
    ("confidence_signal", "confidence signal"),
)


class _Stores:
    """The stores the learner-state block reads from.

    Installed once by the application lifespan (or by a test) via
    :func:`init`. A store that was never installed simply contributes no
    sections.
    """

    def __init__(self) -> None:
        self.learning_store: Any = None
        self.learning_planner: Any = None
        self.interview_store: Any = None


_stores = _Stores()


def init(
    *,
    learning_store: Any = None,
    learning_planner: Any = None,
    interview_store: Any = None,
) -> None:
    """Install the stores :func:`build_learner_state_block` reads from.

    Called once at startup. Passing nothing clears every store, which is
    what a test teardown wants.
    """
    _stores.learning_store = learning_store
    _stores.learning_planner = learning_planner
    _stores.interview_store = interview_store


def _read(store: Any, accessor: str, **kwargs: Any) -> Any:
    """Call one store accessor, degrading to ``None`` on any failure.

    A missing store, a missing accessor, and a raising read are all
    treated the same way: the caller skips the section. "Where an
    accessor does not exist, skip that section rather than inventing
    data."
    """
    if store is None:
        return None
    method = getattr(store, accessor, None)
    if not callable(method):
        return None
    try:
        return method(**kwargs)
    except Exception:
        logger.warning(
            "learner state accessor %s failed", accessor, exc_info=True
        )
        return None


def _rows(payload: Any) -> list[dict]:
    """Coerce an accessor payload into a list of dicts.

    Some accessors return a bare list; the others wrap theirs under a
    well-known key (``weak_areas``, ``points``). Anything else is treated
    as "no data" rather than guessed at.
    """
    if isinstance(payload, dict):
        payload = payload.get("weak_areas") or payload.get("points") or []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _weak_areas_section(
    user_id: str,
    topic_id: str | None,
    track: str | None,
) -> tuple[str, list[str]]:
    """Section (a): weak areas plus topic/track mastery.

    Returns the rendered section and the weak topic ids, which section
    (b) uses as its objective scope when the caller named no topic.
    """
    areas = _rows(
        _read(_stores.learning_store, "get_weak_areas", user_id=user_id, limit=10)
    )
    if topic_id:
        areas = [
            area for area in areas
            if str(area.get("topic_id") or "") == topic_id
        ]
    areas = [area for area in areas if int(area.get("attempts") or 0) > 0]
    # Weakest first, then most-attempted, so the ordering is stable even
    # when a store returns its rows in another order.
    areas.sort(
        key=lambda area: (
            float(area.get("mastery_score") or 0.0),
            float(area.get("accuracy") or 0.0),
            -int(area.get("attempts") or 0),
        )
    )

    lines: list[str] = []
    for area in areas[:_MAX_WEAK_AREAS]:
        lines.append(
            f"{str(area.get('topic_id') or 'unknown')}: "
            f"mastery {float(area.get('mastery_score') or 0.0):.2f}, "
            f"accuracy {_pct(area.get('accuracy'))} over "
            f"{int(area.get('attempts') or 0)} attempt(s), "
            f"{int(area.get('due_count') or 0)} due"
        )

    if track:
        # Track-level mastery lives in the planner's competency scores;
        # without it the per-topic lines still stand on their own.
        for entry in _rows(
            _read(
                _stores.learning_planner,
                "list_competencies",
                user_id=user_id,
                limit=50,
            )
        ):
            if (
                str(entry.get("competency_type") or "") != "track"
                or str(entry.get("track") or "").strip().lower() != track.strip().lower()
            ):
                continue
            score = entry.get("score")
            if isinstance(score, (int, float)):
                lines.append(f"{track} track mastery: {float(score):.0f}/100")
            break

    if not lines:
        return "", []
    return (
        _prompt_section(
            "Weak areas and mastery (weakest first; target these)",
            lines,
        ),
        [str(area.get("topic_id") or "") for area in areas[:_MAX_WEAK_AREAS]],
    )


def _objectives_section(
    user_id: str,
    topic_id: str | None,
    weak_topic_ids: list[str],
) -> str:
    """Section (b): current learning objectives with their accuracy."""
    scope = [topic_id] if topic_id else [tid for tid in weak_topic_ids if tid]
    if not scope:
        return ""
    ranked = _rows(
        _read(
            _stores.learning_store,
            "select_next_objectives",
            user_id=user_id,
            topic_ids=scope,
            limit=_MAX_OBJECTIVES,
        )
    )
    lines: list[str] = []
    for row in ranked:
        objective = str(row.get("learning_objective") or "").strip()
        if not objective:
            continue
        lines.append(
            f"{objective} (accuracy {_pct(row.get('accuracy'))} over "
            f"{int(row.get('attempts') or 0)} attempt(s))"
        )
    if not lines:
        return ""
    return _prompt_section(
        "Current learning objectives (weakest first; probe these)",
        lines,
    )


def _error_patterns_section(user_id: str, topic_id: str | None) -> str:
    """Section (c): top ranked error patterns, deduped by error type."""
    ranked = _rows(
        _read(
            _stores.learning_store,
            "list_error_patterns",
            user_id=user_id,
            topic_id=topic_id,
            limit=10,
            min_count=1,
        )
    )
    lines: list[str] = []
    seen: set[str] = set()
    for row in ranked:
        error_type = str(row.get("error_type") or "").strip()
        if not error_type or error_type.lower() in seen:
            continue
        seen.add(error_type.lower())
        count = int(row.get("count") or 1)
        tally = f" (seen {count}x)" if count > 1 else ""
        explanation = " ".join(
            str(row.get("explanation") or "").split()
        )[:_EXPLANATION_CHARS]
        detail = f": {explanation}" if explanation else ""
        lines.append(f"{error_type}{tally}{detail}")
        if len(lines) >= _MAX_ERROR_PATTERNS:
            break
    if not lines:
        return ""
    return _prompt_section(
        "Recurring error patterns (most frequent first; correct these)",
        lines,
    )


def _interview_section(user_id: str, track: str | None) -> str:
    """Section (d): interview weak dimensions from recent reports."""
    points = _rows(
        _read(_stores.interview_store, "get_trends", user_id=user_id, limit=5)
    )
    if track:
        points = [
            point for point in points
            if str(point.get("track") or "").strip().lower() == track.strip().lower()
        ]
    if not points:
        return ""

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for point in points:
        rubric = point.get("rubric_averages")
        if not isinstance(rubric, dict):
            continue
        for key, _label in _INTERVIEW_DIMENSIONS:
            value = rubric.get(key)
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    averages = {
        key: totals[key] / counts[key] for key in totals if counts.get(key)
    }
    if not averages:
        return ""

    weakest = sorted(averages.items(), key=lambda item: item[1])
    labels = dict(_INTERVIEW_DIMENSIONS)
    lines = [
        f"{labels[key]}: {value:.1f}/5" for key, value in weakest[:_MAX_INTERVIEW_DIMENSIONS]
    ]
    return _prompt_section(
        "Interview weak dimensions (recent average, weakest first)",
        lines,
    )


def _safe_section(
    render: Callable[[], Any],
    default: Any,
) -> Any:
    """Run one section builder, degrading to ``default`` on any failure."""
    try:
        return render()
    except Exception:
        logger.warning("learner state section failed", exc_info=True)
        return default


def build_learner_state_block(
    user_id: str,
    *,
    topic_id: str | None = None,
    track: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Gather the learner's state into one bounded, fenced prompt block.

    Sections are gathered in priority order — weak areas and mastery,
    current objectives with their accuracy, ranked error patterns,
    interview weak dimensions — and the walk stops at the character
    budget, so a small budget keeps the earlier, higher-priority
    sections. Returns ``""`` (no block at all) for an empty user id, no
    data, or a non-positive budget. The function never raises.
    """
    uid = str(user_id or "").strip()
    if not uid:
        return ""
    try:
        budget = DEFAULT_MAX_CHARS if max_chars is None else int(max_chars)
    except (TypeError, ValueError):
        budget = DEFAULT_MAX_CHARS
    if budget <= 0:
        return ""

    topic = str(topic_id or "").strip() or None
    track_name = str(track or "").strip() or None

    weak = _safe_section(
        lambda: _weak_areas_section(uid, topic, track_name), ("", [])
    )
    weak_section, weak_topic_ids = weak if isinstance(weak, tuple) else ("", [])
    candidates = [
        weak_section,
        _safe_section(lambda: _objectives_section(uid, topic, weak_topic_ids), ""),
        _safe_section(lambda: _error_patterns_section(uid, topic), ""),
        _safe_section(lambda: _interview_section(uid, track_name), ""),
    ]

    sections: list[str] = []
    used = 0
    for rendered in candidates:
        if not rendered:
            continue
        projected = used + (2 if sections else 0) + len(rendered)
        if projected > budget:
            break
        sections.append(rendered)
        used = projected
    if not sections:
        return ""

    body = "\n\n".join(sections)
    return untrusted_block(LEARNER_STATE_LABEL, body, budget)
