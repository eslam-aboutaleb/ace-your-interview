"""Goal-driven plan preview shaping (issue #6).

Pure, deterministic functions behind ``POST /api/journeys/plan-preview``:
depth shapes how much of the source outline a plan commits to, weekly
hours drive a pacing estimate, and active FSRS cards on the source topic
mark objectives as review obligations rather than uncovered ground. The
preview persists nothing; the learner edits the syllabus client-side and
commits it through ``POST /api/journeys`` with the echoed provenance.
"""

from __future__ import annotations

import math
from typing import Any

#: Depth -> (max modules, max objectives per module). ``standard`` mirrors
#: the caps journey creation already ships; ``deep`` lifts the module cap
#: to the creation ceiling; ``survey`` is a deliberate quick-scan cut.
DEPTH_CAPS: dict[str, tuple[int, int]] = {
    "survey": (4, 3),
    "standard": (12, 8),
    "deep": (200, 8),
}

VALID_DEPTHS = tuple(DEPTH_CAPS)

#: Planning heuristic: one objective ≈ 30 minutes of study + practice.
MINUTES_PER_OBJECTIVE = 30

COVERAGE_UNCOVERED = "uncovered"
COVERAGE_REVIEW = "review_suggested"

# Edited-plan validation bounds (commit path).
MAX_PLAN_PHASES = 12
MAX_PLAN_MODULES = 200
MAX_OBJECTIVES_PER_MODULE = 16
MAX_TITLE_LENGTH = 120
MAX_DESCRIPTION_LENGTH = 500
VALID_OBJECTIVE_KINDS = ("concept", "practice", "project")


class PlanValidationError(ValueError):
    """Raised when a client-edited plan cannot be committed."""


def shape_plan_for_depth(plan: dict[str, Any], depth: str) -> dict[str, Any]:
    """Trim a baseline plan to the depth's caps (modules, objectives)."""
    if depth not in DEPTH_CAPS:
        raise PlanValidationError(f"depth must be one of {', '.join(VALID_DEPTHS)}")
    max_modules, max_objectives = DEPTH_CAPS[depth]
    shaped_phases: list[dict[str, Any]] = []
    remaining = max_modules
    for phase in plan.get("phases", []):
        modules = []
        for module in phase.get("modules", []):
            if remaining <= 0:
                break
            modules.append(
                {
                    "title": module.get("title", ""),
                    "description": module.get("description", ""),
                    "objectives": [
                        {
                            "title": obj.get("title", ""),
                            "kind": obj.get("kind", "concept"),
                            "source_section": obj.get("source_section", ""),
                        }
                        for obj in module.get("objectives", [])[:max_objectives]
                    ],
                }
            )
            remaining -= 1
        if modules:
            shaped_phases.append(
                {
                    "title": phase.get("title", ""),
                    "description": phase.get("description", ""),
                    "modules": modules,
                }
            )
    return {"phases": shaped_phases}


def mark_review_obligations(
    plan: dict[str, Any], *, has_active_cards: bool
) -> tuple[dict[str, Any], int]:
    """Annotate each objective with its coverage state.

    FSRS cards are topic-granular (no per-section linkage exists yet), so
    when the source topic has active review cards every objective is marked
    ``review_suggested`` rather than ``uncovered`` - the learner has seen
    this material before and owes retention work on it.
    """
    review_count = 0
    for phase in plan.get("phases", []):
        for module in phase.get("modules", []):
            for objective in module.get("objectives", []):
                coverage = COVERAGE_REVIEW if has_active_cards else COVERAGE_UNCOVERED
                objective["coverage"] = coverage
                review_count += 1 if coverage == COVERAGE_REVIEW else 0
    return plan, review_count


def plan_stats(plan: dict[str, Any], *, available_hours_per_week: int) -> dict[str, int]:
    module_count = 0
    objective_count = 0
    review_count = 0
    for phase in plan.get("phases", []):
        for module in phase.get("modules", []):
            module_count += 1
            for objective in module.get("objectives", []):
                objective_count += 1
                if objective.get("coverage") == COVERAGE_REVIEW:
                    review_count += 1
    hours_needed = objective_count * MINUTES_PER_OBJECTIVE / 60.0
    estimated_weeks = max(1, math.ceil(hours_needed / max(1, available_hours_per_week)))
    return {
        "module_count": module_count,
        "objective_count": objective_count,
        "review_objective_count": review_count,
        "estimated_weeks": estimated_weeks,
    }


def validate_edited_plan(plan: Any) -> dict[str, Any]:
    """Validate a client-edited plan for commitment.

    The store re-stamps stable content-hash IDs on commit, so the edited
    plan carries titles/kinds/sections only - never client-chosen IDs.
    """
    if not isinstance(plan, dict):
        raise PlanValidationError("plan must be an object")
    phases = plan.get("phases")
    if not isinstance(phases, list) or not phases:
        raise PlanValidationError("plan.phases must be a non-empty list")
    if len(phases) > MAX_PLAN_PHASES:
        raise PlanValidationError(f"plan is limited to {MAX_PLAN_PHASES} phases")

    total_modules = 0
    total_objectives = 0
    cleaned_phases: list[dict[str, Any]] = []
    for phase in phases:
        if not isinstance(phase, dict):
            raise PlanValidationError("each phase must be an object")
        title = str(phase.get("title", "")).strip()
        if not title or len(title) > MAX_TITLE_LENGTH:
            raise PlanValidationError(
                f"phase titles must be 1-{MAX_TITLE_LENGTH} characters"
            )
        modules = phase.get("modules")
        if not isinstance(modules, list) or not modules:
            raise PlanValidationError(f"phase {title!r} must list at least one module")
        cleaned_modules: list[dict[str, Any]] = []
        for module in modules:
            if not isinstance(module, dict):
                raise PlanValidationError("each module must be an object")
            mtitle = str(module.get("title", "")).strip()
            if not mtitle or len(mtitle) > MAX_TITLE_LENGTH:
                raise PlanValidationError(
                    f"module titles must be 1-{MAX_TITLE_LENGTH} characters"
                )
            objectives = module.get("objectives")
            if not isinstance(objectives, list) or not objectives:
                raise PlanValidationError(f"module {mtitle!r} must list at least one objective")
            if len(objectives) > MAX_OBJECTIVES_PER_MODULE:
                raise PlanValidationError(
                    f"module {mtitle!r} is limited to {MAX_OBJECTIVES_PER_MODULE} objectives"
                )
            cleaned_objectives: list[dict[str, Any]] = []
            for objective in objectives:
                if not isinstance(objective, dict):
                    raise PlanValidationError("each objective must be an object")
                otitle = str(objective.get("title", "")).strip()
                if not otitle or len(otitle) > MAX_TITLE_LENGTH:
                    raise PlanValidationError(
                        f"objective titles must be 1-{MAX_TITLE_LENGTH} characters"
                    )
                kind = str(objective.get("kind", "concept") or "concept").strip().lower()
                if kind not in VALID_OBJECTIVE_KINDS:
                    raise PlanValidationError(
                        f"objective kind must be one of {', '.join(VALID_OBJECTIVE_KINDS)}"
                    )
                cleaned_objectives.append(
                    {
                        "title": otitle,
                        "kind": kind,
                        "source_section": str(objective.get("source_section", ""))[:MAX_TITLE_LENGTH],
                    }
                )
            total_objectives += len(cleaned_objectives)
            cleaned_modules.append(
                {
                    "title": mtitle,
                    "description": str(module.get("description", ""))[:MAX_DESCRIPTION_LENGTH],
                    "objectives": cleaned_objectives,
                }
            )
            total_modules += 1
            if total_modules > MAX_PLAN_MODULES:
                raise PlanValidationError(f"plan is limited to {MAX_PLAN_MODULES} modules")
        cleaned_phases.append(
            {
                "title": title,
                "description": str(phase.get("description", ""))[:MAX_DESCRIPTION_LENGTH],
                "modules": cleaned_modules,
            }
        )
    if total_objectives == 0:
        raise PlanValidationError("plan must contain at least one objective")
    return {"phases": cleaned_phases}
