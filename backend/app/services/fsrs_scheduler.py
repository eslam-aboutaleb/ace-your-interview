"""Facade over the py-fsrs Scheduler (FSRS-4.5).

Every call into the ``fsrs`` library goes through this module so that library
API drift is contained to a single file. The facade works with plain dicts
(the ``fsrs_cards`` row shape) instead of library objects, so the rest of the
application never imports ``fsrs`` directly.

Counters that the library does not track on ``fsrs.Card`` (``reps``,
``lapses``, ``scheduled_days``, ``elapsed_days``) are maintained here and
persisted by :mod:`app.services.learning_store`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fsrs import Card as FSRSCard
from fsrs import Rating as FSRSRating
from fsrs import Scheduler as FSRSScheduler
from fsrs import State as FSRSState

LEECH_LAPSE_THRESHOLD = 5

RATING_AGAIN = "again"
RATING_HARD = "hard"
RATING_GOOD = "good"
RATING_EASY = "easy"
RATINGS = (RATING_AGAIN, RATING_HARD, RATING_GOOD, RATING_EASY)

STATE_NEW = "new"
STATE_LEARNING = "learning"
STATE_REVIEW = "review"
STATE_RELEARNING = "relearning"

_RATING_TO_FSRS: dict[str, FSRSRating] = {
    RATING_AGAIN: FSRSRating.Again,
    RATING_HARD: FSRSRating.Hard,
    RATING_GOOD: FSRSRating.Good,
    RATING_EASY: FSRSRating.Easy,
}
_FSRS_TO_RATING: dict[FSRSRating, str] = {
    value: key for key, value in _RATING_TO_FSRS.items()
}

_STATE_TO_FSRS: dict[str, FSRSState] = {
    STATE_LEARNING: FSRSState.Learning,
    STATE_REVIEW: FSRSState.Review,
    STATE_RELEARNING: FSRSState.Relearning,
}
_FSRS_TO_STATE: dict[FSRSState, str] = {
    value: key for key, value in _STATE_TO_FSRS.items()
}

# FSRS forgetting-curve constants (DECAY = -0.5, FACTOR = 0.9 ** (1/DECAY) - 1)
_FSRS_DECAY = -0.5
_FSRS_FACTOR = 0.9 ** (1 / _FSRS_DECAY) - 1

_scheduler: FSRSScheduler | None = None


def _get_scheduler() -> FSRSScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = FSRSScheduler()
    return _scheduler


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _from_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _normalise_now(now: datetime | None) -> datetime:
    moment = now if now is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _infer_learning_step(due: datetime, last_review: datetime | None) -> int:
    """Infer the FSRS learning step from a card's due interval.

    The facade always uses the default learning steps (1 min, 10 min), so a
    Learning-state card due at or beyond the second step is on step 1. The
    threshold sits between the Hard interval (midpoint of the two steps) and
    the Good interval (the second step itself).
    """
    if last_review is None:
        return 0
    scheduler = _get_scheduler()
    if len(scheduler.learning_steps) < 2:
        return 0
    remaining_seconds = (due - last_review).total_seconds()
    second_step_seconds = scheduler.learning_steps[1].total_seconds()
    threshold = second_step_seconds * 0.75
    return 1 if remaining_seconds >= threshold else 0


def create_card(now: datetime | None = None) -> dict[str, Any]:
    """Create a brand-new FSRS card in serialized (dict) form."""
    moment = _normalise_now(now)
    card = FSRSCard(due=moment)
    return card_to_dict(card)


def card_to_dict(card: FSRSCard) -> dict[str, Any]:
    """Serialize an ``fsrs.Card`` into the ``fsrs_cards`` dict shape."""
    state = _FSRS_TO_STATE.get(card.state, STATE_LEARNING)
    if card.stability is None and card.last_review is None:
        state = STATE_NEW
    return {
        "state": state,
        "stability": card.stability,
        "difficulty": card.difficulty,
        "due_at": _to_iso(card.due),
        "last_review_at": _to_iso(card.last_review) if card.last_review else None,
        "reps": 0,
        "lapses": 0,
        "scheduled_days": 0,
        "elapsed_days": 0,
    }


def card_from_dict(data: dict[str, Any]) -> FSRSCard:
    """Rebuild an ``fsrs.Card`` from the ``fsrs_cards`` dict shape."""
    state_raw = str(data.get("state") or STATE_NEW)
    stability_raw = data.get("stability")
    difficulty_raw = data.get("difficulty")
    stability = float(stability_raw) if stability_raw is not None else None
    difficulty = float(difficulty_raw) if difficulty_raw is not None else None

    if state_raw == STATE_NEW or (stability is None and difficulty is None):
        fsrs_state = FSRSState.Learning
        step = 0
    else:
        fsrs_state = _STATE_TO_FSRS.get(state_raw, FSRSState.Learning)
        if fsrs_state == FSRSState.Learning:
            due = _from_iso(str(data["due_at"]))
            last_review_raw = data.get("last_review_at")
            last_review = (
                _from_iso(str(last_review_raw)) if last_review_raw else None
            )
            step = _infer_learning_step(due, last_review)
        else:
            step = 0

    return FSRSCard(
        state=fsrs_state,
        step=step,
        stability=stability,
        difficulty=difficulty,
        due=_from_iso(str(data["due_at"])),
        last_review=(
            _from_iso(str(data["last_review_at"]))
            if data.get("last_review_at")
            else None
        ),
    )


def review(
    card: dict[str, Any],
    rating: str,
    now: datetime,
    duration_ms: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Review a card with a rating and return ``(new_card, log_entry)``.

    ``card`` is the serialized ``fsrs_cards`` dict shape; both return values
    use the same plain-dict shape so callers can persist them directly.
    """
    if rating not in _RATING_TO_FSRS:
        raise ValueError(f"unknown FSRS rating: {rating!r}")
    moment = _normalise_now(now)
    fsrs_card = card_from_dict(card)
    previous_lapses = int(card.get("lapses") or 0)
    previous_reps = int(card.get("reps") or 0)

    reviewed, _log = _get_scheduler().review_card(
        fsrs_card,
        _RATING_TO_FSRS[rating],
        moment,
        int(duration_ms),
    )

    new_card = card_to_dict(reviewed)
    new_card["reps"] = previous_reps + 1
    new_card["lapses"] = previous_lapses + (1 if rating == RATING_AGAIN else 0)
    if fsrs_card.last_review is not None:
        new_card["elapsed_days"] = max(
            0, (moment - fsrs_card.last_review).days
        )
    else:
        new_card["elapsed_days"] = 0
    new_card["scheduled_days"] = max(0, (reviewed.due - moment).days)

    log_entry = {
        "rating": rating,
        "state": new_card["state"],
        "review_duration_ms": int(duration_ms),
        "scheduled_days": new_card["scheduled_days"],
        "elapsed_days": new_card["elapsed_days"],
        "created_at": _to_iso(moment),
    }
    return new_card, log_entry


def is_leech(card: dict[str, Any]) -> bool:
    """A card is a leech once it has accumulated ``LEECH_LAPSE_THRESHOLD`` lapses."""
    return int(card.get("lapses") or 0) >= LEECH_LAPSE_THRESHOLD


def retention_estimate(
    card: dict[str, Any], now: datetime | None = None
) -> float:
    """Estimate the probability of recalling the card right now.

    Uses the FSRS forgetting curve over the card's current stability and the
    time elapsed since its last review. New cards (no stability) return 0.0.
    """
    stability_raw = card.get("stability")
    last_review_raw = card.get("last_review_at")
    if stability_raw is None or not last_review_raw:
        return 0.0
    stability = float(stability_raw)
    if stability <= 0:
        return 0.0
    moment = _normalise_now(now)
    elapsed_days = max(
        0.0, (moment - _from_iso(str(last_review_raw))).total_seconds() / 86400.0
    )
    retrievability = (1.0 + _FSRS_FACTOR * elapsed_days / stability) ** _FSRS_DECAY
    return max(0.0, min(1.0, retrievability))


# ── Store delegation ─────────────────────────────────────────────────────
# The facade also exposes the per-user queries so that callers can treat this
# module as the single entry point for FSRS work. The implementations live on
# LearningStore (which owns the SQLite connection); the facade receives the
# store instance as an argument to avoid a circular import.


def due_cards(store: Any, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Due, non-suspended cards for a user (delegates to the store)."""
    return store.get_due_cards(user_id=user_id, limit=limit)


def forecast(store: Any, user_id: str, days: int = 7) -> dict[str, Any]:
    """Upcoming due-date histogram for a user (delegates to the store)."""
    return store.get_forecast(user_id=user_id, days=days)


def calibration(store: Any, user_id: str) -> dict[str, Any]:
    """Retention calibration for a user (delegates to the store)."""
    return store.get_calibration(user_id=user_id)


__all__ = [
    "LEECH_LAPSE_THRESHOLD",
    "RATINGS",
    "RATING_AGAIN",
    "RATING_HARD",
    "RATING_GOOD",
    "RATING_EASY",
    "STATE_NEW",
    "STATE_LEARNING",
    "STATE_REVIEW",
    "STATE_RELEARNING",
    "calibration",
    "card_from_dict",
    "card_to_dict",
    "create_card",
    "due_cards",
    "forecast",
    "is_leech",
    "retention_estimate",
    "review",
]
